###############################################################
import csv
import os

import joblib
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, OneClassSVM

from fine_tuning_optimized import grid_search_one_class_svm, grid_search_svm, grid_search_elliptic_envelope, \
    grid_search_random_forest, grid_search_knn
from knn import binary_knn_evaluate, binary_knn_train, binary_knn_predict
from elliptic_envelope import elliptic_envelope_train, elliptic_envelope_evaluate, \
    elliptic_envelope_predict
from file_helper_t3 import load_k_fold_results
from random_forest import binary_rf_train, binary_rf_evaluate, binary_rf_predict
from labels_helper import encode_labels
from svm import one_class_svm_evaluate, binary_svm_train, binary_svm_evaluate, one_class_svm_predict, \
    one_class_svm_train, binary_svm_predict
import numpy as np
from pathlib import Path


def project_indices_to_keep_indices(train_idx, test_idx, keep_indices):
    """
    train_idx, test_idx : Listen/Arrays mit Indizes im 'alten' Indexraum (0..N-1)
    keep_indices        : sortiertes Array mit Indizes, die nach dem Deduplizieren übrig bleiben

    Rückgabe:
      train_new, test_new : Arrays mit Indizes im NEUEN Indexraum (0..len(keep_indices)-1)
    """
    keep_indices = np.asarray(keep_indices)
    # Map: alter Index -> neuer Index
    index_map = {old: new for new, old in enumerate(keep_indices)}

    train_new = [index_map[i] for i in train_idx if i in index_map]
    test_new  = [index_map[i] for i in test_idx if i in index_map]

    return np.array(train_new, dtype=int), np.array(test_new, dtype=int)



#indices not restored
#splits ds based on indices!
#timeestamps should be a list of all timestamps fot the dataset
def split_training_and_test(ds, labels, timestamps, train_indices_fold, test_indices_fold):

    # FEATURES AUFTEILEN
    X_train = ds[train_indices_fold]
    X_test = ds[test_indices_fold]

    # LABELS AUFTEILEN
    y_train = labels[train_indices_fold]
    y_test = labels[test_indices_fold]

    T_test = timestamps[test_indices_fold]  #t_test[index] will be same element as x_test[index] and x_label[index]!
    T_train = timestamps[train_indices_fold]
    return X_train, X_test, y_train, y_test, T_train, T_test
#only return data from keep indices?


def execute_fold_for_experiments_def(
        fold_idx,
        binary_numeric_labels,
        timestamps,
        train_indices,
        test_indices,
        classifier,
        prefix,
        scenario,
        feature_dir="datasets",
        keep_indices=None,
):
    """Train mit Gridsearch usw., optional nur auf deduplizierten Samples (keep_indices)."""

    feature_dir = Path(feature_dir)
    feat_path = feature_dir / f"{prefix}_features_fold{fold_idx}.npy"

    ds_full = np.load(feat_path)

    # Falls keep_indices gesetzt: Features, Labels, Timestamps reduzieren
    if keep_indices is not None:
        ds = ds_full[keep_indices]
        labels_used = binary_numeric_labels[keep_indices]
        timestamps_used = timestamps[keep_indices]

        # UND: Folds in den neuen Indexraum projizieren
        train_indices, test_indices = project_indices_to_keep_indices(
            train_indices, test_indices, keep_indices
        )
    else:
        ds = ds_full
        labels_used = binary_numeric_labels
        timestamps_used = timestamps

    print(f"Executing {classifier} for fold {fold_idx} in Scenario {scenario} "
          f"(dir={feature_dir}, prefix={prefix}).")

    X_train, X_test, y_train, y_test, t_train, t_test = split_training_and_test(
        ds, labels_used, timestamps_used, train_indices, test_indices
    )

    f1 = 0.0

    if classifier == "ocsvm":
        base = OneClassSVM()
        best_model = grid_search_one_class_svm(
            base, X_train, y_train, X_test, y_test, scoring_metric="accuracy"
        )
        joblib.dump(best_model, "models/ocsvm.joblib")
        roc_auc, precision, recall = one_class_svm_evaluate(X_test, y_test)

    elif classifier == "bsvm":
        base = SVC(probability=True, random_state=42)
        best_model = grid_search_svm(
            base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro"
        )
        joblib.dump(best_model, "models/bsvm.joblib")
        roc_auc, precision, recall, f1 = binary_svm_evaluate(X_test, y_test)

    elif classifier == "ee":
        base = EllipticEnvelope()
        best_model = grid_search_elliptic_envelope(
            base, X_train, y_train, X_test, y_test, scoring_metric="accuracy"
        )
        joblib.dump(best_model, "models/ee.joblib")
        roc_auc, precision, recall = elliptic_envelope_evaluate(X_test, y_test)

    elif classifier == "rf":
        base = RandomForestClassifier(random_state=42, n_jobs=1)
        best_model = grid_search_random_forest(
            base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro"
        )
        joblib.dump(best_model, "models/brf.joblib")
        roc_auc, precision, recall, f1 = binary_rf_evaluate(X_test, y_test)

    elif classifier == "knn":
        base = KNeighborsClassifier(n_jobs=1)
        best_model = grid_search_knn(
            base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro"
        )
        joblib.dump(best_model, "models/bknn.joblib")
        roc_auc, precision, recall, f1 = binary_knn_evaluate(X_test, y_test)

    else:
        print("wrong classifier choice")
        return 0.0, 0.0, 0.0, 0.0

    return roc_auc, precision, recall, f1



def execute_scenario_for_experiments_def(
        global_label_encoder,
        classifier,
        k,
        prefix,
        scenario,
        feature_dir="datasets",
        keep_indices=None,
        labels_path=None,
        timestamps_path=None,
        folds_path_template="k_fold_results/k_fold_s{scenario}_{prefix}.json",
):
    """
    - feature_dir: Verzeichnis, in dem *_features_fold*.npy liegen (z.B. für Task d–f).
    - keep_indices: optional Array mit Indizes, die aus dem ORIGINAL-Datensatz übrig bleiben sollen.
    - labels_path / timestamps_path: Pfade zur *vollen* Labels- und Timestamp-Datei,
      aus denen wir bei Bedarf mit keep_indices subsetten.
    """

    if labels_path is None:
        labels_path = f"datasets/{prefix}_labels.npy"
    if timestamps_path is None:
        timestamps_path = f"datasets/{prefix}_timestamps.npy"

    labels_full = np.load(labels_path)
    timestamps_full = np.load(timestamps_path, allow_pickle=True)

    # Falls dedupliziert werden soll: Labels/Timestamps reduzieren
    if keep_indices is not None:
        labels = labels_full[keep_indices]
        timestamps = timestamps_full[keep_indices]
    else:
        labels = labels_full
        timestamps = timestamps_full

    # K-fold-Indices laden (auf Original-Indexraum)
    folds_path = folds_path_template.format(scenario=scenario, prefix=prefix)
    train_indices_all, test_indices_all = load_k_fold_results(folds_path)

    # Falls dedupliziert: alle Folds in neuen Indexraum projizieren
    if keep_indices is not None:
        new_train_all = []
        new_test_all = []
        for tr_old, te_old in zip(train_indices_all, test_indices_all):
            tr_new, te_new = project_indices_to_keep_indices(tr_old, te_old, keep_indices)
            new_train_all.append(tr_new)
            new_test_all.append(te_new)
        train_indices_all = new_train_all
        test_indices_all = new_test_all

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels = np.where(numeric_labels == 0, 0, 1)

    precision_all_folds = []
    recall_all_folds = []

    for fold_idx in range(k):
        roc_auc, precision, recall, f1 = execute_fold_for_experiments_def(
            fold_idx,
            binary_numeric_labels,
            timestamps,
            train_indices_all[fold_idx],
            test_indices_all[fold_idx],
            classifier=classifier,
            prefix=prefix,
            scenario=scenario,
            feature_dir=feature_dir,
            keep_indices=keep_indices,
        )
        precision_all_folds.append(precision)
        recall_all_folds.append(recall)

    return precision_all_folds, recall_all_folds



def execute_experiments_abc(global_label_encoder, k):
    """
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on dataset RAW, suitable for task a to c.
    """


    #onky run the valid scenarios
    def valid_scenarios_for(clf_name: str):
        # OCSVM & EE are only defined for Scenario 1
        if clf_name in ("ocsvm", "ee"):
            return [1]
        # BSVM, RF, kNN are defined for Scenarios 2 & 3
        elif clf_name in ("bsvm", "rf", "knn"):
            return [2, 3]
        else:
            return []

    def run_one(clf_name, scen):
        print(f"\n[classifiers] Running {clf_name} on Scenario {scen} (RAW)\n")

        precisions, recalls = execute_scenario_for_experiments(
            global_label_encoder,
            classifier=clf_name,
            k=k,
            prefix="raw",  # RAW features as required here
            scenario=scen,
        )

        # Convert to plain lists in case they are NumPy arrays
        precisions = list(precisions)
        recalls = list(recalls)

        # One file per classifier+scenario (RAW explicitly in the filename)
        out_path = os.path.join("results", f"{clf_name}_scenario{scen}_raw.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header is optional; comment this out if you don't want it
            writer.writerow(["fold", "precision", "recall"])

            for fold_idx, (p, r) in enumerate(zip(precisions, recalls)):
                writer.writerow([fold_idx, p, r])

        print(f"[classifiers] Saved results to {out_path}")

    # Always run all classifiers with all their valid scenarios
    classifiers_to_run = ["ocsvm", "bsvm", "ee", "rf", "knn"]

    for clf_name in classifiers_to_run:
        scenarios_for_clf = valid_scenarios_for(clf_name)
        for scen in scenarios_for_clf:
            run_one(clf_name, scen)


    return


def execute_experiments_def(global_label_encoder, k):

    #this needs to

    #for each loaded fold remove deduplicates and according indices from fold and training part of fold!


    return





