import csv
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from handling_re_bytes_integrated import get_keep_indices_from_fold0
from use_classifiers import execute_scenario_error_overlap


def create_venn_diagram_data(prediction_errors_scenario):
    #inter_1_2_3 = | A ∩ B ∩ C |
    #inter_1_2 = | A ∩ B | - | A ∩ B ∩ C | (so “pair-only”, excludes triple)
    #only_1 = | A \ B \ C |

    overlap_1_2_3 = overlap_1_2 = overlap_2_3 = overlap_1_3 = 0
    only_1 = only_2 = only_3 = 0

    # We assume all lists are the same length
    num_datapoints = len(prediction_errors_scenario[0])

    for i in range(num_datapoints):
        # Assign to variables to make the if-statements easier to read
        m1 = prediction_errors_scenario[0][i]
        m2 = prediction_errors_scenario[1][i]
        m3 = prediction_errors_scenario[2][i]

        # Fix 2: Order matters! Check the most complex overlap first.
        # Use == 1 to check if it's an error

        if m1 == 1 and m2 == 1 and m3 == 1:
            overlap_1_2_3 += 1

        # Now check double overlaps (which exclude the triple overlap)
        elif m1 == 1 and m2 == 1:
            overlap_1_2 += 1
        elif m2 == 1 and m3 == 1:
            overlap_2_3 += 1
        elif m1 == 1 and m3 == 1:
            overlap_1_3 += 1

        # Finally check single errors (which exclude all overlaps)
        elif m1 == 1:
            only_1 += 1
        elif m2 == 1:
            only_2 += 1
        elif m3 == 1:
            only_3 += 1

    return (overlap_1_2_3, overlap_1_2, overlap_2_3, overlap_1_3,
            only_1, only_2, only_3)


def create_venn_diagram_data_optimized(prediction_errors_scenario):
    #sanity checks
    assert len(prediction_errors_scenario) == 3
    L = len(prediction_errors_scenario[0])
    assert all(len(x) == L for x in prediction_errors_scenario)
    # Convert list of 0/1s into sets of indices where error == 1
    # Example: {0, 5, 9} means errors occurred at these positions
    s1 = {i for i, val in enumerate(prediction_errors_scenario[0]) if val == 1}
    s2 = {i for i, val in enumerate(prediction_errors_scenario[1]) if val == 1}
    s3 = {i for i, val in enumerate(prediction_errors_scenario[2]) if val == 1}

    # Use Set Intersections to find overlaps
    inter_1_2_3 = len(s1 & s2 & s3)
    inter_1_2 = len(s1 & s2) - inter_1_2_3
    inter_2_3 = len(s2 & s3) - inter_1_2_3
    inter_1_3 = len(s1 & s3) - inter_1_2_3

    # Purely only in one model
    only_1 = len(s1 - s2 - s3)
    only_2 = len(s2 - s1 - s3)
    only_3 = len(s3 - s1 - s2)

    return inter_1_2_3, inter_1_2, inter_2_3, inter_1_3, only_1, only_2, only_3




def run_scenarios_for_feature_type(
    k: int,
    global_label_encoder,
    prefix: str,                # "raw" or "re" (for execute_scenario_rt)
    keep_indices: int = 0,
    param: int = 0,
):
    # scenario mapping
    scenario_models = {
        1: ["ocsvm", "lof", "ee"],
        2: ["rf", "knn", "bsvm"],
        3: ["rf", "knn", "bsvm"],
    }

    if prefix=="re":
        fn_prefix="re15"
    else:
        fn_prefix="raw"
    for scenario, models in scenario_models.items():

        prediction_errors_scenario=[]
        for clf in models:
            #returns list with predictions for whole dataset and list of original labels for whole dataset
            prediction_errors = execute_scenario_error_overlap(
                global_label_encoder=global_label_encoder,
                classifier=clf,
                k=k,
                prefix=prefix,
                scenario=scenario,
                keep_indices=keep_indices,
                param=param,
            )
            prediction_errors_scenario.append(prediction_errors)

        inter_1_2_3, inter_1_2, inter_2_3, inter_1_3, only_1, only_2, only_3=create_venn_diagram_data_optimized(prediction_errors_scenario)

        #store one csv file for prefix (raw/15) and scenario:
        # Write CSV for this (prefix, scenario)

        csv_path = f"results/error_overlaps_{fn_prefix}_s{scenario}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "triple_1_2_3",
                "pair_1_2",
                "pair_2_3",
                "pair_1_3",
                "only_1",
                "only_2",
                "only_3",
            ])
            writer.writerow([
                inter_1_2_3,
                inter_1_2,
                inter_2_3,
                inter_1_3,
                only_1,
                only_2,
                only_3,
            ])
        print(f"[OK] wrote {csv_path}")

    return

def plot_venn_from_csv(prefix: str, scenario: int):
    csv_path = f"results/error_overlaps_{prefix}_s{scenario}.csv"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # Extract values (convert from str → int)
    triple = int(row["triple_1_2_3"])
    pair_12 = int(row["pair_1_2"])
    pair_23 = int(row["pair_2_3"])
    pair_13 = int(row["pair_1_3"])
    only_1 = int(row["only_1"])
    only_2 = int(row["only_2"])
    only_3 = int(row["only_3"])

    # Order required by venn3:
    # (only A, only B, A∩B, only C, A∩C, B∩C, A∩B∩C)
    subsets = (
        only_1,
        only_2,
        pair_12,
        only_3,
        pair_13,
        pair_23,
        triple,
    )

    # Model labels depend on scenario
    if scenario == 1:
        labels = ("OCSVM", "LOF", "EE")
    else:
        labels = ("RF", "KNN", "BSVM")

    plt.figure(figsize=(6, 6))
    venn3(subsets=subsets, set_labels=labels)

    plt.title(f"Venn Diagram – {prefix.upper()} – Scenario {scenario}")
    plt.tight_layout()

    out_path = f"results/venn_{prefix}_s{scenario}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[OK] saved {out_path}")
    return


def all_error_overlaps(k: int, global_label_encoder):
    """
    Runs for RAW and for RE15, measures classification error overlaps
      - Scenario 1: ocsvm, lof, ee
      - Scenario 2 & 3: rf, knn, bsvm
    """
    scenarios=[1,2,3]
    Path("results").mkdir(exist_ok=True)

    # ---------------- RAW ----------------

    run_scenarios_for_feature_type(
        k=k,
        global_label_encoder=global_label_encoder,
        prefix="raw",
        keep_indices=0,
        param=0,
    )
    # ---------------- RE15 ----------------
    keep_indices = get_keep_indices_from_fold0("datasets/re_bytes_15", "re15")

    run_scenarios_for_feature_type(
        k=k,
        global_label_encoder=global_label_encoder,
        prefix="re",
        keep_indices=keep_indices,
        param=15,
    )

    #create plots from csv files
    for scen in scenarios:
        plot_venn_from_csv("raw",scen)
        plot_venn_from_csv("re15", scen)
    return