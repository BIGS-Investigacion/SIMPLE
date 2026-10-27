import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

# In this framework the algorithm and the model are always the same.
ALGORITHM = "SIMPLE"
MODEL = "default"


def is_minimized(metric, mode="auto"):
    """
    Decide whether the selection metric is minimized (loss) or maximized
    (accuracy, etc.).

    mode="auto": minimized if the metric name contains "loss".
    mode="min" / "max": forces the direction.
    """
    if mode == "min":
        return True
    if mode == "max":
        return False
    return "loss" in metric.lower()


def pick_best_idx(grouped_series, minimize):
    """idxmin if minimizing, idxmax if maximizing."""
    return grouped_series.idxmin() if minimize else grouped_series.idxmax()


# Data Loading

EPOCH_FILENAMES = ("epochs.jsonl", "epoch.jsonl")


def load_records(sweep_dirs):
    """
    Takes one or more sweep folders, e.g.
        root/MyDataset/Sweep_SIMPLE_LODO/
    and searches them for every <Hash>/ subfolder containing
    results.json + epochs.jsonl. Merges results.json with each line of
    epochs.jsonl. Returns a flat list of records, one per (run, epoch).
    """
    records = []

    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        if not sweep_dir.is_dir():
            print(f"[WARN] {sweep_dir} is not a directory, skipping.")
            continue

        for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
            results_path = run_dir / "results.json"
            epoch_path = next(
                (run_dir / n for n in EPOCH_FILENAMES if (run_dir / n).is_file()),
                None,
            )

            if not results_path.is_file() or epoch_path is None:
                print(f"[WARN] Missing files in {run_dir}, skipping.")
                continue

            with results_path.open(encoding="utf-8") as f:
                results = json.load(f)

            with epoch_path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        epoch_row = json.loads(line)
                    except json.JSONDecodeError:
                        print(
                            f"[WARN] Could not parse line {line_no} "
                            f"of {epoch_path}, skipping."
                        )
                        continue

                    r = {**results, **epoch_row}
                    r["_run_hash"] = run_dir.name
                    # If results.json has no "dataset", it is inferred from
                    # the path: <Dataset>/<Sweep>/<Hash>
                    r["_dataset_dir"] = run_dir.parent.parent.name
                    records.append(r)

    return records


def build_lodo_dataframe(records, metric="acc"):
    """
    Converts leave-two-out records into observations of the LODO protocol,
    for ALL possible test domains (instead of a fixed one).

    Each record has domainbed_test_envs=[a, b]: two domains excluded from
    training. That single record can play two different roles:

        role 1: val=a, test=b
        role 2: val=b, test=a

    Therefore, each record generates TWO rows in the DataFrame (one for each
    choice of test domain), reusing the same env{a}_in_{metric} /
    env{b}_in_{metric} metrics that are already in the record.

    trial_seed DOES define the experiment (kept as an independent axis).
    seed does NOT define the experiment (it is metadata, not used for grouping).

    Since results.json contains neither "algorithm" nor "model", the
    ALGORITHM and MODEL constants defined at the top of the file are used.
    """
    rows = []
    val_col = f"val_{metric}"
    test_col = f"test_{metric}"

    for r in records:
        test_envs = r.get("domainbed_test_envs", [])

        if len(test_envs) != 2:
            continue

        a, b = test_envs
        for val_env, test_env in ((a, b), (b, a)):
            val_key = f"env{val_env}_in_{metric}"
            test_key = f"env{test_env}_in_{metric}"

            if val_key not in r or test_key not in r:
                continue

            rows.append({
                "dataset": r.get("dataset", r.get("_dataset_dir")),
                "algorithm": ALGORITHM,
                "model": MODEL,

                # seed does NOT define the experiment.
                "seed": r.get("seed"),

                # trial_seed DOES define the experiment: each trial_seed is
                # treated as an independent experiment throughout the pipeline.
                "trial_seed": r.get("trial_seed"),

                "hparams_seed": r.get("hparam_seed", r.get("hparams_seed")),
                "step": r.get("step"),

                "val_env": val_env,
                "test_env": test_env,

                val_col: r[val_key],
                test_col: r[test_key],

                "output_dir": r.get("_run_hash"),
            })

    df = pd.DataFrame(rows)

    # Losses can be NaN/inf if training diverges. Those rows are dropped so
    # that idxmin/idxmax neither fail nor select invalid values.
    if not df.empty:
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.dropna(subset=[val_col, test_col]).copy()
        df[val_col] = df[val_col].astype(float)
        df[test_col] = df[test_col].astype(float)

    return df


# ---------------------------------------------------------------------------
# MODEL SELECTION (unchanged from the original)
# ---------------------------------------------------------------------------

def select_best_step_per_run(df, metric="acc", minimize=False):
    """
    For each individual run (dataset, algorithm, model, trial_seed,
    hparams_seed, val_env, test_env), keeps the epoch that reaches the
    BEST value of the validation metric (maximum for accuracy, minimum for
    loss; "oracle early-stopping" style selection within the run itself).

    This collapses the "step"/epoch axis, leaving a single (val, test) point
    for each combination of run + hparams_seed + val_env.
    """
    val_col = f"val_{metric}"

    group_cols = [
        "dataset", "algorithm", "model",
        "trial_seed", "hparams_seed",
        "val_env", "test_env",
    ]

    idx = pick_best_idx(df.groupby(group_cols)[val_col], minimize)
    best_step = df.loc[idx].reset_index(drop=True)
    best_step = best_step.rename(columns={"step": "best_step"})
    return best_step


def compute_hparam_selection(best_step_df, metric="acc", minimize=False):
    """
    Selects the winning hparams_seed SEPARATELY for each (dataset, algorithm,
    model, trial_seed, test_env).

    The score of each hparams_seed is the average (over the val_envs
    available for that test_env) of the best validation value reached in any
    of the epochs (see select_best_step_per_run). The winner is the one with
    the highest score if maximizing, or the lowest score if minimizing
    (loss).

    Only hparams_seeds that have results for the COMPLETE set of val_envs
    expected for that (trial_seed, test_env) are compared; if an hparams_seed
    has fewer val_envs than others, it is discarded to keep the comparison
    fair.
    """
    val_col = f"val_{metric}"
    mean_val_col = f"mean_{val_col}"
    std_val_col = f"std_{val_col}"

    config_cols = [
        "dataset", "algorithm", "model",
        "trial_seed", "test_env", "hparams_seed",
    ]
    group_keys = ["dataset", "algorithm", "model", "trial_seed", "test_env"]

    val_summary = (
        best_step_df.groupby(config_cols)
        .agg(**{
            mean_val_col: (val_col, "mean"),
            std_val_col: (val_col, "std"),
            "n_val_envs": ("val_env", "nunique"),
            "val_envs_used": ("val_env", lambda s: tuple(sorted(set(s)))),
        })
        .reset_index()
    )

    # "Expected" set of val_envs for each (trial_seed, test_env): the union
    # of val_envs seen across all candidate hparams_seeds.
    expected = (
        val_summary
        .groupby(group_keys)["val_envs_used"]
        .apply(lambda s: tuple(sorted(set().union(*s))))
        .rename("expected_val_envs")
        .reset_index()
    )
    val_summary = val_summary.merge(expected, on=group_keys, how="left")

    val_summary = val_summary[
        val_summary["val_envs_used"] == val_summary["expected_val_envs"]
    ].drop(columns="expected_val_envs").copy()

    idx = pick_best_idx(val_summary.groupby(group_keys)[mean_val_col], minimize)
    best = val_summary.loc[idx].reset_index(drop=True)
    return best, val_summary


def compute_test_results(best_step_df, best_hparams, metric="acc"):
    """
    Retrieves, for each (dataset, algorithm, model, trial_seed, test_env),
    the test results corresponding to the winning hparams_seed, averaging
    over the val_envs that took part in that selection.

    Each trial_seed remains completely separate: it is never mixed with other
    trial_seeds at any point in the pipeline.
    """
    val_col = f"val_{metric}"
    test_col = f"test_{metric}"
    mean_val_col = f"mean_{val_col}"
    mean_test_col = f"mean_{test_col}"
    std_test_col = f"std_{test_col}"

    keys = ["dataset", "algorithm", "model", "trial_seed", "test_env", "hparams_seed"]
    group_keys = ["dataset", "algorithm", "model", "trial_seed", "test_env"]

    selected = best_step_df.merge(
        best_hparams[keys],
        on=keys,
        how="inner",
    )

    test_summary = (
        selected
        .groupby(group_keys)
        .agg(**{
            mean_test_col: (test_col, "mean"),
            std_test_col: (test_col, "std"),
            "n_test_runs": (test_col, "count"),
            "n_val_envs": ("val_env", "nunique"),
            "val_envs_used": ("val_env", lambda s: tuple(sorted(set(s)))),
            "best_steps_used": ("best_step", lambda s: tuple(sorted(set(s)))),
            "hparams_seed": ("hparams_seed", "first"),
        })
        .reset_index()
    )

    test_summary = test_summary.merge(
        best_hparams[group_keys + [mean_val_col]],
        on=group_keys,
        how="left",
    )

    return test_summary


def _safe_component(value):
    """Converts a value into a safe folder name (no path separators)."""
    s = str(value)
    s = re.sub(r"[\\/]+", "_", s)
    s = s.strip().strip(".")
    return s or "unknown"


def save_results_by_group(test_summary, out_dir, suffix=""):
    """
    Saves test_summary to out_dir/dataset/algorithm{suffix}/model/best.csv,
    one file for each (dataset, algorithm, model) combination. Folders are
    created automatically if they do not exist.
    """
    saved = []
    for (dataset, algorithm, model), sub in test_summary.groupby(
        ["dataset", "algorithm", "model"], dropna=False
    ):
        folder = os.path.join(
            out_dir,
            _safe_component(dataset),
            _safe_component(f"{algorithm}{suffix}"),
            _safe_component(model),
        )
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "best.csv")
        sub.to_csv(path, index=False)
        saved.append(path)
    return saved


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "sweep_dirs",
        nargs="+",
        help="Sweep folder(s) containing the <Hash>/ folders, "
             "e.g. root/BreastHist/Sweep_SIMPLE_LODO/"
    )

    parser.add_argument(
        "--metric",
        default="acc",
        help=(
            "Metric to use as the selection and reporting criterion "
            "(default: acc). E.g.: acc, loss. Metrics whose name "
            "contains 'loss' are minimized automatically."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "max", "min"],
        default="auto",
        help=(
            "Optimization direction. 'auto' (default) minimizes if the "
            "metric contains 'loss' and maximizes otherwise; 'max' and "
            "'min' force the direction."
        ),
    )

    parser.add_argument(
        "--test-envs",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Restricts the output to these test domains. "
            "By default, ALL observed test domains are computed."
        ),
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output folder (optional). Results are saved to "
            "<out>/<dataset>/<algorithm>/<model>/best.csv; folders are "
            "created automatically if they do not exist."
        ),
    )

    parser.add_argument(
        "--suffix",
        default="",
        help=(
            "Optional suffix appended to the algorithm folder name. "
            "E.g.: --suffix _LP saves to "
            "<out>/<dataset>/<algorithm>_LP/<model>/best.csv"
        ),
    )

    args = parser.parse_args()

    metric = args.metric
    minimize = is_minimized(metric, args.mode)
    direction = "minimum" if minimize else "maximum"

    val_col = f"val_{metric}"
    test_col = f"test_{metric}"
    mean_val_col = f"mean_{val_col}"
    std_val_col = f"std_{val_col}"
    mean_test_col = f"mean_{test_col}"
    std_test_col = f"std_{test_col}"

    print(f"[INFO] Metric: {metric} | selection criterion: {direction}")

    # Load records (results.json + epoch.jsonl merged)

    records = load_records(args.sweep_dirs)

    if not records:
        print("No records found.")
        return

    # Convert leave-two-out -> LODO protocol, for ALL possible test domains

    df = build_lodo_dataframe(records, metric=metric)

    if df.empty:
        print(
            "No valid records found for the LODO protocol."
        )
        return

    if args.test_envs is not None:
        df = df[df["test_env"].isin(args.test_envs)].copy()
        if df.empty:
            print("No data for the requested --test-envs.")
            return

    # For each run, keep the epoch that gives the best value of the
    # validation metric (collapses the step/epoch axis)

    best_step_df = select_best_step_per_run(df, metric=metric, minimize=minimize)

    # Select hparams_seed using ONLY validation, separately
    # for each trial_seed and each test_env

    best_hparams, val_summary = compute_hparam_selection(
        best_step_df, metric=metric, minimize=minimize
    )

    print(
        f"\n=== Best hparams_seed per trial_seed and test domain "
        f"(selected using validation ONLY, {direction} per epoch) ===\n"
    )

    cols = [
        "dataset",
        "algorithm",
        "model",
        "trial_seed",
        "test_env",
        "hparams_seed",
        mean_val_col,
        std_val_col,
        "n_val_envs",
        "val_envs_used",
    ]

    print(
        best_hparams[cols]
        .sort_values([
            "dataset",
            "algorithm",
            "model",
            "trial_seed",
            "test_env",
        ])
        .to_string(index=False)
    )

    # Evaluate that configuration on the test domain(s),
    # separately for each trial_seed

    test_summary = compute_test_results(
        best_step_df,
        best_hparams,
        metric=metric,
    )

    print(
        "\n=== Final result per trial_seed and test domain "
        "(each trial_seed is an independent experiment) ===\n"
    )

    cols_test = [
        "dataset",
        "algorithm",
        "model",
        "trial_seed",
        "test_env",
        "hparams_seed",
        mean_val_col,
        mean_test_col,
        std_test_col,
        "n_test_runs",
        "n_val_envs",
        "val_envs_used",
        "best_steps_used",
    ]

    test_summary_sorted = test_summary[cols_test].sort_values([
        "dataset",
        "algorithm",
        "model",
        "trial_seed",
        "test_env",
    ])

    print(test_summary_sorted.to_string(index=False))

    # Pivoted view: test domains as columns, for quick
    # comparison across trial_seeds

    pivot = test_summary.pivot_table(
        index=["dataset", "algorithm", "model", "trial_seed"],
        columns="test_env",
        values=mean_test_col,
    )
    pivot.columns = [f"test_env_{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    print(
        "\n=== Pivoted view: mean_test_{metric} by test domain "
        "(columns) and trial_seed (rows) ===\n".format(metric=metric)
    )
    print(pivot.to_string(index=False))

    # Save: out/dataset/algorithm/model/best.csv

    if args.out:
        saved = save_results_by_group(
            test_summary_sorted, args.out, suffix=args.suffix
        )
        print(f"\nResults saved to {len(saved)} file(s):")
        for p in saved:
            print(f"  - {p}")


if __name__ == "__main__":
    main()