import numpy as np
import pandas as pd
import os
import argparse
import json
import collections
from miscellaneous.domainbed import datasets
from miscellaneous.domainbed.lib import misc, reporting
from miscellaneous.domainbed.lib.query import Q

def load_records(path):
    records = []
    for i, subdir in list(enumerate(os.listdir(path))):
        results_path = os.path.join(path, subdir, "results.json")
        try:
            with open(results_path, "r") as f:
                records.append(json.load(f))
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records):
    result = collections.defaultdict(lambda: [])
    for r in records:
            group = (r["trial_seed"], r["dataset"], r["domainbed_test_env"])
            result[group].append(r)
    return Q([{"trial_seed": t, "dataset": d, "test_env": e,
        "records": Q(r)} for (t,d,e),r in result.items()])

def IIDAccuracySelectionMethod(records):

    return records.argmax('best_val_acc')['valbest_test_acc']

def format_mean(data, latex):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "X"
    mean =  np.mean(list(data))
    err = np.std(list(data) / np.sqrt(len(data)))
    if latex:
        return mean, err, "{:.1f} $\\pm$ {:.1f}".format(mean, err)
    else:
        return mean, err, "{:.1f} +/- {:.1f}".format(mean, err)

def print_trials(results, domain_names=None, latex=False):
    """
    Print all trials per domain neatly with proper domain names.
    """
    if domain_names is None:
        domain_names = [f"Domain {i+1}" for i in range(len(results))]
    
    print("Leave-One-Domain-Out Trials:")
    print("-" * 60)
    
    for name, trials in zip(domain_names, results):
        trials_str = ", ".join(f"{t:.2f}" for t in trials)
        print(f"{name:15}: [{trials_str}]")
    
    print("-" * 60)

def print_table(table, header_text, row_labels, col_labels, colwidth=10,
    latex=True):
    """Pretty-print a 2D array of data, optionally with row/col labels"""
    print("")

    if latex:
        num_cols = len(table[0])
        print("\\begin{center}")
        print("\\adjustbox{max width=\\textwidth}{%")
        print("\\begin{tabular}{l" + "c" * num_cols + "}")
        print("\\toprule")
    else:
        print("--------", header_text)

    for row, label in zip(table, row_labels):
        row.insert(0, label)

    if latex:
        col_labels = ["\\textbf{" + str(col_label).replace("%", "\\%") + "}"
            for col_label in col_labels]
    table.insert(0, col_labels)

    for r, row in enumerate(table):
        misc.print_row(row, colwidth=colwidth, latex=latex)
        if latex and r == 0:
            print("\\midrule")
    if latex:
        print("\\bottomrule")
        print("\\end{tabular}}")
        print("\\end{center}")


def main():

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    records = load_records(args.input_dir)
    dataset_names = Q(records).select("dataset").unique().sorted()

    # Only 1 datasaet per sweep allowed
    if len(dataset_names) > 1:
        raise NotImplementedError
    else:
        dataset = dataset_names[0]

    # get sweep_acc value (best test acc given a (trial_seed, dataset, test_env) pair)
    grouped_records = get_grouped_records(records)
    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": IIDAccuracySelectionMethod(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    # Get dataset table
    test_envs = range(datasets.num_environments(dataset))
    table = [[None for _ in [*test_envs, "Avg"]]]
    means = []
    trial_accs_list = []
    for j, test_env in enumerate(test_envs):
        trial_accs = (grouped_records
            .filter_equals(
                "dataset, test_env",
                (dataset, test_env)
            ).select("sweep_acc"))
        mean, err, table[0][j] = format_mean(trial_accs, False)
        means.append(mean)
        trial_accs_list.append(trial_accs)

    print_trials(trial_accs_list, datasets.get_dataset_class(dataset).ENVIRONMENTS, latex=False)

    if None in means:
        table[0][-1] = "X"
    else:
        table[0][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = [
            "Algorithm",
            *datasets.get_dataset_class(dataset).ENVIRONMENTS,
            "Avg"
        ]
    header_text = (f"Dataset: {dataset}, "
        f"model selection method: {'Train-validation'}")
    print_table(table, header_text, ['SIMPLE'], list(col_labels),
        colwidth=20, latex=False)

    # Get Averages Table
    table = [[None for _ in [*dataset_names, "Avg"]]]
    means = []
    for j, dataset in enumerate(dataset_names):
        trial_averages = (grouped_records
            .filter_equals("dataset", (dataset))
            .group("trial_seed")
            .map(lambda trial_seed, group:
                group.select("sweep_acc").mean()
            )
        )
        mean, err, table[0][j] = format_mean(trial_averages, False)
        means.append(mean)
    if None in means:
        table[0][-1] = "X"
    else:
        table[0][-1] = "{:.1f}".format(sum(means) / len(means))

    col_labels = ["Algorithm", *dataset_names, "Avg"]
    header_text = f"Averages, model selection method: {'Train-Validation'}"
    print_table(table, header_text, ['SIMPLE'], col_labels, colwidth=25,
        latex=False)

if __name__=='__main__':
    main()

