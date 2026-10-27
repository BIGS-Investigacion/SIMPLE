# LODO validation protocol

These scripts reproduce the leave-one-domain-out (LODO) model-selection protocol of DomainBed, while avoiding runs that would never be selected. Instead of training every combination of 1 and 2 held-out domains, the sweep only trains the combinations of **2** held-out domains. Each of those runs can then play two roles:

- held-out pair `[a, b]` → `val = a, test = b`
- held-out pair `[a, b]` → `val = b, test = a`

so a single run provides validation and test results for both choices of test domain.

## Pipeline overview

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | `sweep_lodoval` | – | `<OUTPUT_ROOT>/<run_hash>/{results.json, epochs.jsonl}` |
| 1 | `1-get_best_lodoval` | sweep folders | `best.csv` with the best `hparams_seed` per (dataset, trial_seed, test_env) |
| 2 | `2-run_best_hparams` | `best.csv` | final runs (local or Slurm) |
| 3 | `3-get_final_results` | final logs | one row per run, at the best epoch |
| 4 | `4-get_table` | results CSV | LaTeX table |

Every run (in the sweep and in the final runs) is executed by `spec_lodoval.py` and writes two files in its output folder:

- `results.json`: written at the end of the run (it marks a finished run). Contains the run metadata (`dataset`, `domainbed_test_envs`, `trial_seed`, `hparam_seed`, `seed`, ...).
- `epochs.jsonl`: one JSON line per epoch, with the metrics `env{j}_in_{metric}` / `env{j}_out_{metric}` of every environment `j`.

---

## 0. Run the sweep

`sweep_lodoval.py` generates, for every dataset, trial and `hparams_seed`, **all unordered pairs of held-out environments** (`n_envs` choose 2; 6 pairs for 4 environments), and launches `spec_lodoval.py` for each one.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sweep_lodoval launch \
    --domainbed_dataset "$Dataset" \
    --pretrain_model_list "$POOL_DIR" \
    --n_hparams_from "$N_hparams_from" \
    --n_hparams "$N_hparams" \
    --n_envs "$N_envs" \
    --input_dir "$INPUT_DIR" \
    --infer_dir "$INFER_DIR" \
    --output_dir "$OUTPUT_ROOT" \
    --seed $Seed \
    --n_trials $N_trials \
    --command_launcher local \
    --skip_confirmation
```

Number of runs: `n_trials × n_datasets × C(n_envs, 2) × (n_hparams − n_hparams_from)`.

Hyperparameters (`ensemble_weight`, `child_weight`, `confidence_weight`, `batch_size`, `ensemble_lr`, `weight_decay`) are sampled from `hparams_seed` (`random.seed(hparams_seed)`). `clf_lr` is fixed to 1. The sampled `batch_size` also selects the inference folder: `<infer_dir>/<batch_size>`.

Expected layout of the output (one sweep call per dataset): `<root>/<Dataset>/Sweep_SIMPLE_LODO/<run_hash>/{results.json, epochs.jsonl}`. For example:

```bash
OUTPUT_ROOT="output/$Dataset/Sweep_SIMPLE_LODO"
```

Notes:

- `<run_hash>` is the md5 of the run arguments, so the same configuration always lands in the same folder.
- `launch` only launches jobs that are `Not launched`. Jobs that are `Incomplete` (output folder exists but no `done` file) are **not** relaunched; remove them first with `python -m sweep_lodoval delete_incomplete ...` (same arguments) and launch again.
- `--command_launcher` accepts `local` (serial), `multi_gpu` (experimental) and `dummy` (only prints the commands). Each command is built with `CUDA_VISIBLE_DEVICES=0` hard-coded in `sweep_lodoval.py`, so `multi_gpu` will not spread the runs across GPUs unless you change that.
- `--seed` is accepted but not used by `make_args_list`; the per-run seed is derived from `(dataset, test_envs, hparams_seed, trial_seed)`.

## 1. Get the best hyperparameters

For every (dataset, trial_seed, test_env), this step picks the `hparams_seed` with the best validation score, using **validation data only**:

1. Each run is expanded into the two LODO roles (`val = a, test = b` and `val = b, test = a`), using `env{val}_in_{metric}` as validation and `env{test}_in_{metric}` as test.
2. Within each run, keep the epoch with the best validation metric.
3. Average that value over the available validation domains.
4. Choose the `hparams_seed` with the best average. Only `hparams_seed`s that have results for the complete set of validation domains are compared.

```bash
python 1-get_best_lodoval.py \
    "output/$Dataset/Sweep_SIMPLE_LODO" \
    --metric acc \
    --out Best_Hparams
```

Notes:

- The positional arguments are **sweep folders** (the ones containing the `<run_hash>/` folders), not globs. You can pass several of them.
- Metrics whose name contains `loss` are minimized; the rest are maximized. Use `--mode min|max` to force the direction.
- `--test-envs` restricts the output to some test domains (by default, all of them are computed).
- Each `trial_seed` is treated as an independent experiment and is never mixed with the others. `seed` is only metadata.
- Rows with `NaN`/`inf` values (e.g. diverged training) are discarded.
- Results are written to `Best_Hparams/<dataset>/SIMPLE<suffix>/default/best.csv`, one file per (dataset, algorithm, model). `--suffix` (default empty) is appended to the algorithm folder name.
- Besides the CSV, the script prints the selected `hparams_seed`, the final test results and a pivoted view (test domains as columns, `trial_seed` as rows).

## 2. Final runs with the best hyperparameters

Train again with the selected hyperparameters. The hyperparameters are regenerated from `hparams_seed` with the same sequence of `random` calls as in `sweep_lodoval.py`, so **`sample_hparams` in `2-run_best_hparams` must be kept in sync with `make_args_list` in `sweep_lodoval.py`**.

Two modes are available (`--mode`):

- `single` (default): one run per row, trained on **all domains except `test_env`** (`--domainbed_test_env <test_env>`). This is the mode you want for the final results. It requires `--holdout-fraction` between 0 and 1, so that an `out` split exists to choose the best epoch in step 3.
- `pair` (kept only for reproducibility): re-runs the leave-two-out protocol of the sweep, i.e. it trains again with the held-out pair `[val_env, test_env]` for every `val_env` in the `val_envs_used` column of the CSV. It requires `--holdout-fraction 0` (or omitting it) and does not produce anything new for the final results (those pairs were already trained in the sweep). Runs are de-duplicated when two rows request the same pair with the same `trial_seed` and `hparams_seed`.

Use the same pool of models (`--pool-dir` / `--pool-list`) and inference folder (`--infer-root`) as in the sweep, and always pass `--input-dir` explicitly (its default is a machine-specific path).

### On Slurm (one `sbatch` per run)

Run it from the login node:

```bash
python 2-run_best_hparams.py "$csv_path" \
    --mode single \
    --name "$TrialName" \
    --input-dir "$INPUT_DIR" \
    --pool-dir modelpool_list \
    --pool-list "$POOL_LIST" \
    --infer-root "$INFER_DIR" \
    --output-root Final_Results \
    --holdout-fraction 0.2 \
    --skip-existing \
    --slurm \
    --job-prefix SIMPLE_BH \
    --partition "$PARTITION" \
    --gres gpu:1 \
    --cpus-per-task 8 \
    --mem 8G \
    --time 3-00:00:00
```

Inside each job, `--setup-cmd` is executed first (default: load the profile, `conda activate SIMPLE`), then `python -m spec_lodoval ...`. Use `--sbatch-extra` for any other `sbatch` option (e.g. `--sbatch-extra="--qos=normal"`).

### Locally (no Slurm)

Runs are executed sequentially with the current Python interpreter (`--python` to change it).

```bash
python 2-run_best_hparams.py "$csv_path" \
    --mode single \
    --name "$TrialName" \
    --input-dir "$INPUT_DIR" \
    --pool-dir modelpool_list \
    --pool-list "$POOL_LIST" \
    --infer-root "$INFER_DIR" \
    --output-root Final_Results \
    --cuda-device 0 \
    --holdout-fraction 0.2 \
    --skip-existing
```

Notes:

- `"$csv_path"` is the `best.csv` from step 1, e.g. `Best_Hparams/$Dataset/SIMPLE/default/best.csv`.
- Add `--dry-run` to print the commands (or the `sbatch` calls) without running anything.
- `--skip-existing` skips runs whose output directory already contains files (ignoring `slurm-*.out`).
- `--stop-on-error` stops at the first failure (default: keep going and report failures at the end; the exit code is 1 if any run failed).
- Filters (`--datasets`, `--algorithms`, `--models`, `--trial-seeds`, `--test-envs`) restrict which rows of the CSV are launched.
- `--module` (default `spec_lodoval`) is the module run with `python -m`, and `--extra "..."` appends extra arguments to it.
- Outputs go to `<output-root>/<dataset>/<name>/SIMPLE/default/TS<trial_seed>_HP<hparams_seed>/TE<envs>/`. The Slurm logs are written to the same folder (`slurm-<jobid>.out`).

## 3. Get final results

For each finished run (one with `results.json` and `epochs.jsonl`), selects the epoch with the best validation metric (mean of `env{j}_out_{metric}` over the source domains `j != test_env`; the earliest epoch wins ties) and reports the test metrics (`env{test_env}_in_{metric}`) at that epoch. By default the validation metric is `loss`, which is **minimized**.

```bash
python 3-get_final_results.py \
    --data_dir Final_Results \
    --out Final_Results/results.csv
```

Notes:

- `--select_metric` accepts `loss` (default, minimized), or `acc`, `bacc`, `f1`, `f1_pos0`, `f1_pos1`, `auroc` (maximized). The direction is chosen automatically: metrics whose name contains `loss` are minimized, the rest are maximized.
- `--epoch N` forces a fixed epoch instead of the best one on validation.
- `--folders` restricts the search to paths that contain a folder matching one of the given patterns (`fnmatch`), e.g. `--folders "$TrialName"`. By default it searches the whole `--data_dir`.
- `--metrics` chooses which test metrics are reported (default: all of `acc bacc f1 f1_pos0 f1_pos1 auroc loss`). Per-class `test_sensitivity_c<k>` / `test_specificity_c<k>` columns are added when the run logs them.
- Every run must have exactly **one** test environment, so only `single` runs can be collected. Keep `pair` runs out of `--data_dir` (or exclude them with `--folders`), otherwise the script stops with an error.
- If several runs share the same (dataset, trial_seed, test_env), a warning is printed with the repeated rows.
- The output CSV has one row per run, with the columns `dataset`, `trial_seed`, `test_env`, `hparam_seed`, `epoch`, `val_<metric>` and the `test_*` metrics. It does not contain an `algorithm` column; step 4 assumes `SIMPLE`.

## 4. Get summary table

Averages over test domains within each seed, then aggregates over seeds (mean or median ± standard error) and writes a LaTeX table with the best result per dataset in bold. The table reports Acc, BAcc, F1, AUROC, Sens and Spec; sensitivity and specificity are taken from class 1 for binary datasets and averaged over the classes for multiclass ones. Requires `\usepackage{booktabs}` and `\usepackage{multirow}`.

```bash
python 4-get_table.py \
    --csv Final_Results/results.csv \
    --out Final_Results/table.tex
```

Useful options: `--agg mean|median`, `--scale 100` (percentages), `--decimals 1`, `--caption`, `--label`. Without `--out`, the table is printed to the terminal.

If the CSV has no `algorithm` column, it is filled with `SIMPLE` (the "Method" column of the table).