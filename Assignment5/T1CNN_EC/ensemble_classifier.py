import csv
from pathlib import Path

import numpy as np

from use_classifiers import execute_scenario_for_ensemble_classifier
from handling_re_bytes_integrated import get_keep_indices_from_fold0


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
            f"datasets/re_bytes_{param}/re_features_fold{param}",
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
