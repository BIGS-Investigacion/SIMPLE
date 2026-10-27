import argparse
import json
import math
import re
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

METRICS = ["acc", "bacc", "f1", "f1_pos0", "f1_pos1", "auroc", "loss"]
OUT_RE = re.compile(r"^env(\d+)_out_(.+)$")

PER_CLASS_METRICS = ["sensitivity", "specificity"]
CLASSES = None

def detect_classes(rec, te):
    """Classes present in env{te}_in_sensitivity (dict or list)."""
    raw = rec.get(f"env{te}_in_sensitivity")
    if isinstance(raw, dict):
        return sorted(int(k) for k in raw)
    if isinstance(raw, (list, tuple)):
        return list(range(len(raw)))
    return []

def _class_value(x, cls):
    """Value of class `cls` from a dict or list; NaN if it does not exist."""
    if isinstance(x, dict):
        return x.get(str(cls), x.get(cls, float("nan")))
    if isinstance(x, (list, tuple)):
        return x[cls] if cls < len(x) else float("nan")
    return float("nan")

def val_score(rec, select_metric):
    """Mean of env{j}_out_<select_metric> across the source environments."""
    vals = []
    for k, v in rec.items():
        m = OUT_RE.match(k)
        if m and m.group(2) == select_metric:
            if isinstance(v, (int, float)) and not math.isnan(v):
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


def is_minimized(select_metric):
    """Losses are minimized; every other metric (acc, f1, ...) is maximized."""
    return "loss" in select_metric.lower()


def read_record(path, select_metric, epoch=None):
    """Returns (record, val_score).

    If epoch is None, picks the epoch with the best validation score:
    the lowest one if select_metric is a loss, the highest otherwise
    (tie: the earliest wins). If epoch is set, returns that one.
    """
    minimize = is_minimized(select_metric)
    best, best_score = None, None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if epoch is not None:
                if rec.get("epoch") == epoch:
                    return rec, val_score(rec, select_metric)
                continue
            s = val_score(rec, select_metric)
            if s is None:
                continue
            if best_score is None or (s < best_score if minimize else s > best_score):
                best, best_score = rec, s
    if epoch is not None or best is None:
        return None, None
    return best, best_score


def in_allowed_folder(res_path, data_dir, folders):
    """True if any folder in the path (relative to data_dir) matches
    any of the patterns. With no patterns, no filtering is applied."""
    if not folders:
        return True
    parts = res_path.relative_to(data_dir).parts[:-1]
    return any(fnmatch(part, pat) for part in parts for pat in folders)


def get_data(data_dir, metrics, select_metric, epoch=None, folders=None):
    rows = []
    # A finished run has results.json (written at the end) and epochs.jsonl
    for res_path in Path(data_dir).rglob("results.json"):
        if not in_allowed_folder(res_path, Path(data_dir), folders):
            continue
        ep_path = res_path.parent / "epochs.jsonl"
        if not ep_path.exists():
            print(f"[WARN] No epochs.jsonl in {res_path.parent}")
            continue

        with res_path.open("r", encoding="utf-8") as f:
            res = json.load(f)

        # Without joint lodoval, only one environment is excluded
        envs = res["domainbed_test_envs"]
        if len(envs) != 1:
            raise ValueError(f"{res_path}: expected 1 test env but found {envs}")
        te = envs[0]

        rec, vscore = read_record(ep_path, select_metric, epoch)
        if rec is None:
            print(f"[WARN] No valid epoch (missing env*_out_{select_metric} keys?) "
                  f"in {ep_path}")
            continue

        row = {
            "dataset": res["dataset"],
            "trial_seed": res["trial_seed"],
            "test_env": te,
            "hparam_seed": res["hparam_seed"],
            "epoch": rec["epoch"],
            "step": rec["step"],
            f"val_{select_metric}": vscore,
            "run_dir": str(res_path.parent),
        }
        # f1_pos0 / f1_pos1 only exist if the dataset is binary -> NaN if missing
        for m in metrics:
            row[f"test_{m}"] = rec.get(f"env{te}_in_{m}", float("nan"))

        for m in PER_CLASS_METRICS:
            raw = rec.get(f"env{te}_in_{m}")
            for c in detect_classes(rec, te):
                row[f"test_{m}_c{c}"] = _class_value(raw, c)
        rows.append(row)

    if not rows:
        raise ValueError(f"No finished runs found in {data_dir}")
    return pd.DataFrame(rows)


def print_table(df, metrics, select_metric):
    index = ["dataset", "trial_seed", "test_env"]
    per_class = sorted(c for c in df.columns
                    if re.fullmatch(r"test_(sensitivity|specificity)_c\d+", c))
    cols = (["hparam_seed", "epoch", f"val_{select_metric}"]
            + [f"test_{m}" for m in metrics] + per_class)
    table = df.sort_values(index).set_index(index)[cols]

    dups = table.index.duplicated(keep=False)
    if dups.any():
        print(f"WARNING: {dups.sum()} rows with a repeated index (several runs per group):")
        print(table[dups])

    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(table)

    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="output")
    parser.add_argument("--metrics", nargs="+", default=METRICS)
    parser.add_argument("--select_metric", type=str, default="loss", choices=METRICS,
                        help="Validation metric used to choose the epoch "
                             "(default: loss). 'loss' is minimized; the rest "
                             "are maximized")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Forces a fixed epoch (by default, the best on validation)")
    parser.add_argument("--folders", nargs="+", default=None,
                        help="Only searches paths that contain a folder with "
                             "By default, searches all of data_dir")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = get_data(args.data_dir, args.metrics, args.select_metric, args.epoch,
                  args.folders)
    table = print_table(df, args.metrics, args.select_metric)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.reset_index().to_csv(args.out, index=False)
        print(f"CSV saved to {args.out}")


if __name__ == "__main__":
    main()