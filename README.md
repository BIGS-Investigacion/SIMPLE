# Domain Generalization in Computational Pathology: SIMPLE

> Modified version of SIMPLE used to evaluate domain generalization methods in histopathology.
> Built on the [original SIMPLE repository](https://github.com/microsoft/SeqML/tree/main/SIMPLE) and part of the [DomainBed](https://github.com/BIGS-Investigacion/DomainBed) pathology benchmark.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19471815.svg)](https://doi.org/10.5281/zenodo.19471815)

## Overview

This repository provides:

- **SIMPLE**, modified for our experiments on histopathology.
- Support for pathology foundation models. A small model pool of pathology models is in [`modelpool_list/patho_list.txt`](modelpool_list/patho_list.txt) (see [Models](#models)).
- Support for the **unified multi-domain histopathology benchmark**: 16 organ-specific domains from 15 public datasets ([Histo-DG-Benchmark](https://doi.org/10.5281/zenodo.19471815)).
- An **efficient LODO validation protocol** adapted to SIMPLE: [`miscellaneous/domainbed/scripts/lodoval`](miscellaneous/domainbed/scripts/lodoval/readme.md).

## Methods compared

| Method | Where to find it |
|--------|------------------|
| ERM (full fine-tuning) | [BIGS-Investigacion/DomainBed](https://github.com/BIGS-Investigacion/DomainBed) (`ERM`) |
| ERM<sub>LP</sub> (linear probing) | [BIGS-Investigacion/DomainBed](https://github.com/BIGS-Investigacion/DomainBed) (`ERM` with `"linear_probe": true`) |
| ERM<sub>LoRA</sub> | [BIGS-Investigacion/DomainBed](https://github.com/BIGS-Investigacion/DomainBed) (`ERM_LoRA`) |
| MoA (Mixture-of-Adapters) | [BIGS-Investigacion/MoA](https://github.com/BIGS-Investigacion/MoA), modified version |
| SIMPLE | This repository |

The domain-shift analyses ([`dshift_analysis`](https://github.com/BIGS-Investigacion/DomainBed)) are in the DomainBed repository.

## Installation

```bash
git clone https://github.com/BIGS-Investigacion/SIMPLE.git
cd SIMPLE

conda create -n SIMPLE python=3.11.11
conda activate SIMPLE

pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu126

pip install git+https://github.com/mahmoodlab/CONCH.git@141cc09c7d4ff33d8eda562bd75169b457f71a62
pip install git+https://github.com/mahmoodlab/TRIDENT.git@25f15396116c43c0b1feb928d5331c8b2a772dfd

pip install -r requirements.txt
```

**Tested with:** Python 3.11.11, PyTorch 2.7.0, CUDA 12.6, NVIDIA A40 GPU, Rocky Linux 8.8 (x86_64).

## Models

We support foundation models. The models are defined and implemented in [`model_wrappers.py`](model_wrappers.py). To add a new pathology model, define and implement it in this script.

**We do not provide the weights of any model.** Follow the license terms of each model. Some models can need access permission.

There are two ways to get the weights:

- **With [Trident](https://github.com/mahmoodlab/TRIDENT).** Some models are built with Trident, and Trident downloads their weights.
- **Local checkpoint.** Some models are not built with Trident. Download their weights and save them locally. Then give the path to the checkpoint in the model pool list (see below).

See `model_wrappers.py` for the details of each model.

### Model pool

The folder [`modelpool_list`](modelpool_list) has the model pool lists. The file `patho_list.txt` has a small pool of pathology models.

To add a pathology model to a `.txt` file, use this key:

```
(patho)Model:Arg
```

- `Model` is the name of the model, as defined in `model_wrappers.py`.
- `Arg` defines the checkpoint path or specifies the weights. Its value depends on the model. See `model_wrappers.py` for each model.

## Data

The benchmark has **4 organs** (breast, colon, lung, prostate) with **4 domains each**. Each domain comes from a different public dataset. LC25000 is used for both colon and lung, which gives 16 domains from 15 datasets.

| Organ | Domains (original datasets) | Classes |
|-------|-----------------------------|---------|
| Breast | BACH, BRACS, BreakHis, Burnasyan | Normal/Benign, Malignant |
| Colon | Chaoyang, EBHI-Seg, Kather2016, LC25000 | Normal/Benign, Malignant |
| Lung | LC25000, LubLung, LungHist700, UNN-LC | Normal/Benign, Malignant |
| Prostate | Gleason2019, GleasonCNN, CrowdGleason, SICAPv2 | Normal, Gleason 3, Gleason 4, Gleason 5 |

**We do not redistribute the original data.** The [Zenodo record](https://doi.org/10.5281/zenodo.19471815) (CC BY 4.0) contains the code and instructions to rebuild the processed benchmarks from the original sources. Please cite the original dataset papers (their DOIs are listed in the Zenodo record).

1. Download `Histo-DG-Benchmark.rar` from Zenodo.
2. Follow the instructions in each organ folder to download and process the original datasets.
3. Place the result in your data directory, following the DomainBed structure `dataset/domain/class/image.xyz`.

For example, with the data stored in `/home/datasets/BreastHist/BACH/...`, use `--input_dir "/home/datasets/" --dataset "domainbed" --domainbed_dataset "BreastHist".`

## Quick start

To get inferences with an specifyed model pool:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="./pytorch_pretrained_models/"
python -u spec_lodoval.py --save_inference_only --dataset domainbed --domainbed_dataset "$Dataset" --input_dir="$INPUT_DIR" --pretrain_model_list "$POOL_DIR" --infer_dir="$INFER_DIR" --batch_size $Batch_size
```

To train model dispatcher:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="./pytorch_pretrained_models/"
CUDA_VISIBLE_DEVICES=0 python -m spec_lodoval \
  --pretrain_model_list "$POOL_DIR" \
  --dataset domainbed \
  --domainbed_dataset "$Dataset" \
  --input_dir="$INPUT_DIR" \
  --infer_dir="$INFER_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --domainbed_test_env $test_env \
  --batch_size $Batch_size \
  --holdout_fraction 0.2
```

## Reproducing the experiments

The full pipeline has six steps. Details and all options are in [`miscellaneous/domainbed/scripts/lodoval/readme.md`](miscellaneous/domainbed/scripts/lodoval/readme.md).

| Step | Script | Result |
|------|--------|--------|
| 0 | `spec_lodoval.py`  | Saved inferences of the model pool |
| 1 | `sweep_lodoval.py` | Sweep logs (`results.json` and `epochs.jsonl`) |
| 2 | `miscellaneous.domainbed.scripts.lodoval.1-get_best_lodoval` | `best.csv` with the best `hparams_seed` per (dataset, algorithm, model, trial_seed, test_env) |
| 3 | `miscellaneous.domainbed.scripts.lodoval.2-run_best_hparams` | Final runs with the best hparams (local or Slurm) |
| 4 | `miscellaneous.domainbed.scripts.lodoval.3-get_final_results.py` | One row per run, at a fixed step |
| 5 | `miscellaneous.domainbed.scripts.lodoval.4-get_table.py` | LaTeX table |

Instead of training every combination of 1 and 2 held-out domains, the sweep only trains the combinations of **2** held-out domains (`--lodoval`). Each run provides validation and test results for both choices of test domain. Every `trial_seed` is treated as an independent experiment.

To run the inferences, launch `spec.lodoval.py` with `--save_inference_only` and set the model pool with `--pretrain_model_list`. To run the sweep with several batch sizes, launch the script once for each batch size. In our experiments, we use batch sizes of 32, 64, and 128.

Example of launching a sweep:

```bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME="./pytorch_pretrained_models/"

CUDA_VISIBLE_DEVICES=0 python -m sweep_lodoval launch \
  --domainbed_dataset "$Dataset"  \
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

## Citation

If you use this code, please cite our paper:

```bibtex
@article{fernandez2026domainshift,
  title   = {Domain Generalization in Computational Pathology: Analyzing Shifts and Benchmarking Methods},
  author  = {Fernández, Jesús and Pachón, Rafael and Bhushan, Megha and Domínguez, Francisco J. and García, Jorge and Macías, Laura},
  journal = {Journal of Imaging Informatics in Medicine},
  year    = {Under review}
}
```

To cite the benchmark data: Histo-DG-Benchmark, [10.5281/zenodo.19471815](https://doi.org/10.5281/zenodo.19471815).

---