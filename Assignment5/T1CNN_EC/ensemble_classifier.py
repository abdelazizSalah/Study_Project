import numpy as np

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



def execute_scenario_for_ensemble_classifier(global_label_encoder, k):
    """
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on dataset RAW, suitable for task a to c.
    """


    #small helper returns the valid scenarios per model
    def valid_scenarios_for(clf_name: str):
        # OCSVM & EE are only defined for Scenario 1
        if clf_name in ("ocsvm", "ee", "lof"):
            return [1]
        # BSVM, RF, kNN are defined for Scenarios 2 & 3
        elif clf_name in ("bsvm", "rf", "knn"):
            return [2, 3]
        else:
            return []

    #run one scenario for a specific model
    def run_one(clf_name, scen):
        print(f"\n[classifiers] Running {clf_name} on Scenario {scen} (RAW)\n")

        precisions, recalls = execute_scenario_for_experiments(
            global_label_encoder,
            classifier=clf_name,
            k=k,
            prefix="raw",  # RAW features as required here
            scenario=scen,
        )


        precisions = list(precisions)
        recalls = list(recalls)

        # One file per classifier+scenario (RAW explicitly in the filename)
        out_path = os.path.join("results", f"{clf_name}_scenario{scen}_raw.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow(["fold", "precision", "recall"])    #header

            for fold_idx, (p, r) in enumerate(zip(precisions, recalls)):
                writer.writerow([fold_idx, p, r])

        print(f"[classifiers] Saved results to {out_path}")

    # Always run all classifiers with all their valid scenarios
    classifiers_to_run = ["ocsvm", "bsvm", "ee", "rf", "knn", "lof"]

    for clf_name in classifiers_to_run:
        scenarios_for_clf = valid_scenarios_for(clf_name)
        for scen in scenarios_for_clf:
            run_one(clf_name, scen)


    return




#1. compute keep_indices for re_bytes_{param}
#2. when using labels, timestamps or k folds -> deduplicate based on keep_indices
#3. use remaining logic as it is
# execute this for param: 5,10,15
def execute_experiments_def(global_label_encoder, k, param=5):
    """
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on dataset /re_bytes_{param}/re_features_fold{param}, suitable for task d to f.
    """

    #calculate keep_indices for file!

    keep_indices=get_keep_indices_from_fold0(f"datasets/re_bytes_{param}/re_features_fold{param}", f"re{param}")
    print(f"For RE{param} - {len(keep_indices)} datapoints are used.")

    #only run the valid scenarios
    def valid_scenarios_for(clf_name: str):
        # OCSVM & EE are only defined for Scenario 1
        if clf_name in ("ocsvm", "ee", "lof"):
            return [1]
        # BSVM, RF, kNN are defined for Scenarios 2 & 3
        elif clf_name in ("bsvm", "rf", "knn"):
            return [2, 3]
        else:
            return []

    def run_one(clf_name, scen, param):
        print(f"\n[classifiers] Running {clf_name} on Scenario {scen} (RE{param})\n")

        precisions, recalls = execute_scenario_for_experiments(
            global_label_encoder,
            classifier=clf_name,
            k=k,
            prefix="re",  # RAW features as required here
            scenario=scen,
            keep_inidces=keep_indices,
            param=param
        )

        # Convert to plain lists in case they are NumPy arrays
        precisions = list(precisions)
        recalls = list(recalls)

        # One file per classifier+scenario (RAW explicitly in the filename)
        out_path = os.path.join("results", f"{clf_name}_scenario{scen}_re{param}.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["fold", "precision", "recall"])

            for fold_idx, (p, r) in enumerate(zip(precisions, recalls)):
                writer.writerow([fold_idx, p, r])

        print(f"[classifiers] Saved results to {out_path}")

    # Always run all classifiers with all their valid scenarios
    classifiers_to_run = ["ocsvm", "bsvm", "ee", "rf", "knn", "lof"]

    for clf_name in classifiers_to_run:
        scenarios_for_clf = valid_scenarios_for(clf_name)
        for scen in scenarios_for_clf:
            run_one(clf_name, scen, param)


    return
