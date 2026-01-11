import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import plt

from use_classifiers import execute_scenario_for_ensemble_classifier
from handling_re_bytes_integrated import get_keep_indices_from_fold0
import matplotlib.pyplot as plt


#p1,p2,p3 are binary lists
def get_ec_prediction(p1,p2,p3, method, seed: int = 0):
    p1 = np.asarray(p1, dtype=np.int8)
    p2 = np.asarray(p2, dtype=np.int8)
    p3 = np.asarray(p3, dtype=np.int8)

    if not (len(p1) == len(p2) == len(p3)):
        raise ValueError("p1, p2, p3 must have the same length")

    if method == "random":
        # choose 0/1/2 for each sample and take corresponding prediction
        rng = np.random.default_rng(seed)   #numpy generator for random numbers
        choice = rng.integers(0, 3, size=len(p1))  # length n, values in {0,1,2}
        stacked = np.stack([p1, p2, p3], axis=0)  #3D array - row0 - p1, row1 -p2, row 2 -p3
        p = stacked[choice, np.arange(len(p1))]  # pick one row per column, result -> one row


    elif method == "majority":
        # sum is 0..3; majority => at least 2 positives
        p = ((p1 + p2 + p3) >= 2).astype(np.int8)

    elif method == "all":
        # all positives
        p = (p1 & p2 & p3).astype(np.int8)

    else:
        raise ValueError("method must be one of: 'random', 'majority', 'all'")

    return p  # numpy array of 0/1





def run_ec_classifier_on_dataset(method, param, global_label_encoder, k):
    scenarios = [1, 2, 3]

    # determine representation + correct prefix/keep_indices
    if param != 0:
        keep_indices = get_keep_indices_from_fold0(
            f"datasets/re_bytes_{param}/",
            f"re{param}"
        )
        print(f"For RE{param} - {len(keep_indices)} datapoints are used.")
        representation = f"re{param}"
        prefix_for_exec = f"re{param}"
    else:
        keep_indices = []
        representation = "raw"
        prefix_for_exec = "raw"

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summary_path = out_dir / f"ec_summary_{representation}.csv"
    write_header = not summary_path.exists()


    method_names = ["random", "majority", "all"]

    rows = []

    for scen in scenarios:
        multiclass = 0 if scen == 1 else 1

        #returns list in case of "all_methods" being used
        p_list, r_list, f1_list = execute_scenario_for_ensemble_classifier(
            method=method,
            multiclass=multiclass,
            global_label_encoder=global_label_encoder,
            k=k,
            prefix=prefix_for_exec,
            scenario=scen,
            keep_indices=keep_indices,
            param=param
        )


        rows.append((scen, p_list, r_list, f1_list))

        with summary_path.open("a", newline="") as f:
            w = csv.writer(f)

            if write_header:
                if method == "all_methods":
                    header = ["representation", "scenario"]
                    for m in method_names:
                        header += [f"precision_{m}", f"recall_{m}", f"f1_{m}"]
                    w.writerow(header)
                else:
                    w.writerow(["representation", "scenario", "avg_precision", "avg_recall", "avg_f1"])
                write_header = False

            if method == "all_methods":
                row = [representation, scen]
                for i in range(3):
                    row += [f"{p_list[i]:.6f}", f"{r_list[i]:.6f}", f"{f1_list[i]:.6f}"]
                w.writerow(row)
            else:
                w.writerow([representation, scen, f"{p_list[0]:.6f}", f"{r_list[0]:.6f}", f"{f1_list[0]:.6f}"])

    return rows



def run_experiment_ec(method,global_label_encoder, k):
    """
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on dataset RAW, suitable for task a to c.
    """
    prefixes = ["raw", "re5", "re10", "re15"]

    params = [0, 5, 10, 15]
    results = {}
    for i in range(len(prefixes)):
        run_ec_classifier_on_dataset(method, params[i], global_label_encoder,k)


#############plots:


# fixed ensemble method order (must match your CSV columns)
METHODS = ["random", "majority", "all"]
COLORS = ["tab:blue", "tab:orange", "tab:green"]


def barplot_one_representation(csv_path: str | Path, out_dir: str | Path = "results/plots"):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load CSV into DataFrame
    df = pd.read_csv(csv_path)

    # determine representation name
    rep = csv_path.stem.replace("ec_summary_", "")
    if "representation" in df.columns and len(df) > 0:
        rep = str(df.loc[0, "representation"])

    # ensure scenario is numeric and sorted
    df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce")
    df = df.dropna(subset=["scenario"]).sort_values("scenario")
    scenarios = df["scenario"].astype(int).tolist()

    # check if all_methods columns exist
    has_all_methods = all(f"precision_{m}" in df.columns for m in METHODS)

    if has_all_methods:
        col_map = {
            "precision": [f"precision_{m}" for m in METHODS],
            "recall":    [f"recall_{m}" for m in METHODS],
            "f1":        [f"f1_{m}" for m in METHODS],
        }
        legend_labels = METHODS
    else:
        # fallback: single-method results
        col_map = {
            "precision": ["avg_precision"],
            "recall":    ["avg_recall"],
            "f1":        ["avg_f1"],
        }
        legend_labels = ["method"]

    # grouped bar parameters
    x = np.arange(len(scenarios))
    n_bars = len(col_map["precision"])
    width = 0.25 if n_bars == 3 else 0.5

    for metric_name, cols in col_map.items():
        plt.figure(figsize=(7, 4))

        for i, col in enumerate(cols):
            y = pd.to_numeric(df[col], errors="coerce").to_numpy()

            offset = (i - (n_bars - 1) / 2) * width
            color = COLORS[i] if n_bars == 3 else None

            plt.bar(
                x + offset,
                y,
                width=width,
                label=legend_labels[i],
                color=color,
            )

            # value labels
            for xi, yi in zip(x + offset, y):
                if np.isfinite(yi):
                    plt.text(
                        xi,
                        yi + 0.01,
                        f"{yi:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.xticks(x, scenarios)
        plt.ylim(0.0, 1.0)  # FIXED SCALE
        plt.xlabel("Scenario")
        plt.ylabel(metric_name.upper())
        plt.title(f"Ensemble Classifier – {rep} – {metric_name.upper()}")
        plt.legend()
        plt.tight_layout()

        out_path = out_dir / f"ec_{rep}_{metric_name}_bar.png"
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"[OK] Saved {out_path}")


def plot_all_representations_ec(
    results_dir: str | Path = "results",
    out_dir: str | Path = "results/plots",
):
    results_dir = Path(results_dir)
    csv_files = sorted(results_dir.glob("ec_summary_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No files found matching {results_dir / 'ec_summary_*.csv'}"
        )

    for csv_path in csv_files:
        barplot_one_representation(csv_path, out_dir=out_dir)
