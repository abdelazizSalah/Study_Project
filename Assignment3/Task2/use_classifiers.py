###############################################################
import csv
import os

import joblib
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC, OneClassSVM
import pandas as pd
from local_outlier_factor import local_outlier_factor_evaluate, local_outlier_factor_train, local_outlier_factor_predict
from handling_re_bytes_integrated import get_keep_indices_from_fold0
from labels_helper import deduplicate_labels_and_timestamps, deduplicate_folds
from fine_tuning_optimized import grid_search_one_class_svm, grid_search_svm, grid_search_elliptic_envelope, \
    grid_search_random_forest, grid_search_knn, grid_search_lof_parallel
from knn import binary_knn_evaluate, binary_knn_train, binary_knn_predict
from elliptic_envelope import elliptic_envelope_train, elliptic_envelope_evaluate, \
    elliptic_envelope_predict
from file_helper_t3 import load_k_fold_results
from random_forest import binary_rf_train, binary_rf_evaluate, binary_rf_predict
from labels_helper import encode_labels
from svm import one_class_svm_evaluate, binary_svm_train, binary_svm_evaluate, one_class_svm_predict, \
    one_class_svm_train, binary_svm_predict
import numpy as np

#indices not restored
#splits ds based on indices!
#timeestamps should be a list of all timestamps fot the dataset
def split_training_and_test(ds, labels, timestamps, train_indices_fold, test_indices_fold):

    # separate features
    X_train = ds[train_indices_fold]
    X_test = ds[test_indices_fold]

    # separate labels
    y_train = labels[train_indices_fold]
    y_test = labels[test_indices_fold]

    #separate timestamps
    T_test = timestamps[test_indices_fold]  #t_test[index] will be same element as x_test[index] and x_label[index]!
    T_train = timestamps[train_indices_fold]
    return X_train, X_test, y_train, y_test, T_train, T_test


#input: indices for one fold of one scenario!
#ocsvm=1 - ocsvm, 0 - binary svm
def execute_fold(fold_idx, binary_numeric_labels, timestamps,train_indices, test_indices, classifier, prefix, scenario):
    """train, measure result, print timestamp + prediction for test dataset for one fold"""


    ds = np.load(f"datasets/{prefix}_features_fold{fold_idx}.npy")    #fold_idx: 1 to k, files are named 0 to k-1
    print(f"Excuting {classifier} for fold {fold_idx} in Scenario {scenario}.")

    X_train, X_test, y_train, y_test, t_train, t_test=split_training_and_test(ds, binary_numeric_labels, timestamps,train_indices, test_indices)

    #run on small portion of dataset, for demonstration:
    #first_indices = np.arange(0, 1000)
    #last_indices = np.arange(len(X_train) - 1000, len(X_train))
    #selected_indices = np.concatenate((first_indices, last_indices))
    #X_train = X_train[selected_indices]
    #y_train = y_train[selected_indices]

    if classifier=="ocsvm":
        one_class_svm_train(X_train)
        one_class_svm_evaluate(X_test, y_test)  #test data and corresponding labels
        prediction_report=one_class_svm_predict(X_test, t_test) #test data and corresponding timestamps
    elif classifier=="bsvm":   #binary svm (multiple classes in training)
        binary_svm_train(X_train, y_train)
        binary_svm_evaluate(X_test, y_test)
        prediction_report=binary_svm_predict(X_test, t_test)
    elif classifier == "ee":
        elliptic_envelope_train(X_train)
        elliptic_envelope_evaluate(X_test, y_test)  # test data and corresponding labels
        prediction_report = elliptic_envelope_predict(X_test, t_test)  # test data and corresponding timestamps

    elif classifier == "rf":
        binary_rf_train(X_train, y_train)
        binary_rf_evaluate(X_test, y_test)
        prediction_report = binary_rf_predict(X_test, t_test)
    elif classifier == "knn":
        binary_knn_train(X_train, y_train)
        binary_knn_evaluate(X_test, y_test)
        prediction_report = binary_knn_predict(X_test, t_test)
    elif classifier == "lof":
        local_outlier_factor_train(X_train)
        local_outlier_factor_evaluate(X_test, y_test)
        prediction_report=local_outlier_factor_predict(X_test, t_test)
    else:
        print("wrong classifier choice")

    pd.set_option("display.max_rows", None)
    print("\n--- Individual Packet Attack Detection Report ---")
    print(prediction_report)
    return


#train and test indices for each fold ([[][],...]
def execute_scenario(global_label_encoder, classifier, k, prefix, scenario):

    labels=np.load(f"datasets/{prefix}_labels.npy")
    timestamps = np.load(f"datasets/{prefix}_timestamps.npy", allow_pickle=True)

    #load train_indices and test_indices for specific scenario
    #contain indices for all of the k folds
    train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_{prefix}.json")


    numeric_labels = encode_labels(global_label_encoder, labels) #make labels numeric
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack


    for fold_idx in range(k):
        execute_fold(fold_idx, binary_numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],classifier=classifier, prefix=prefix, scenario=scenario)

    return





def execute_fold_for_experiments(fold_idx, binary_numeric_labels, timestamps,
                                 train_indices, test_indices, classifier, prefix, scenario, keep_indices=0, param=0):
    """Train with grid search, save best model, then use *_evaluate for metrics."""

    if param!=0:
        ds = np.load(f"datasets/re_bytes_{param}/re{param}_features_fold{fold_idx}.npy")
        print(f"Executing {classifier} for fold {fold_idx} in Scenario {scenario} for RE{param}.")
    else:
        ds = np.load(f"datasets/{prefix}_features_fold{fold_idx}.npy")
        print(f"Executing {classifier} for fold {fold_idx} in Scenario {scenario} for RAW.")
    #for task d - f


    X_train, X_test, y_train, y_test, t_train, t_test = split_training_and_test(
        ds, binary_numeric_labels, timestamps, train_indices, test_indices
    )

    if classifier == "ocsvm":
        base = OneClassSVM()
        best_model = grid_search_one_class_svm(base, X_train, y_train, X_test, y_test, scoring_metric="accuracy")
        # overwrite the model file used by one_class_svm_evaluate
        joblib.dump(best_model, "models/ocsvm.joblib") # save the best model
        roc_auc, precision, recall = one_class_svm_evaluate(X_test, y_test) # the model will be loaded inside this function.

    elif classifier == "bsvm":
        base = SVC(probability=True, random_state=42)
        best_model = grid_search_svm(base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro")
        joblib.dump(best_model, "models/bsvm.joblib")  # adjust name if different in your code
        precision, recall, f1 = binary_svm_evaluate(X_test, y_test)

    elif classifier == "ee":
        base = EllipticEnvelope()
        # select first 100 samples, and last 100 samples from X_trains for testing only
        X_train_subset = np.concatenate([X_train[:100], X_train[-100:]])
        y_train_subset = np.concatenate([y_train[:100], y_train[-100:]])
        X_test_subset = np.concatenate([X_test[:100], X_test[-100:]])
        y_test_subset = np.concatenate([y_test[:100], y_test[-100:]])
        best_model = grid_search_elliptic_envelope(base, X_train_subset, y_train_subset, X_test_subset, y_test_subset, scoring_metric="accuracy")
        joblib.dump(best_model, "models/ee.joblib")    # adjust filename to match your evaluate()
        roc_auc, precision, recall = elliptic_envelope_evaluate(X_test, y_test)

    elif classifier == "rf":
        base = RandomForestClassifier(random_state=42, n_jobs=1)
        best_model = grid_search_random_forest(base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro")
        joblib.dump(best_model, "models/brf.joblib")   # whatever binary_rf_train uses
        precision, recall, f1 = binary_rf_evaluate(X_test, y_test)

    elif classifier == "knn":
        base = KNeighborsClassifier(n_jobs=1)
        best_model = grid_search_knn(base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro")
        joblib.dump(best_model, "models/bknn.joblib")  # adjust to your filename
        precision, recall, f1 = binary_knn_evaluate(X_test, y_test)
    elif classifier == "lof":
        base = LocalOutlierFactor(novelty=True, n_jobs=1)  #without novelty=True, LOF cannot score unseen samples
        best_model = grid_search_lof_parallel(base, X_train, y_train, X_test, y_test, scoring_metric="accuracy")
        joblib.dump(best_model, "models/lof.joblib")  # adjust to your filename
        roc_auc, precision, recall = local_outlier_factor_evaluate(X_test, y_test)
    else:
        print("wrong classifier choice")
        return 0.0, 0.0, 0.0, 0.0

    return precision, recall



#train and test indices for each fold ([[][],...]
def execute_scenario_for_experiments(global_label_encoder, classifier, k, prefix, scenario, keep_inidces=0, param=0):

    labels=np.load(f"datasets/{prefix}_labels.npy")
    timestamps = np.load(f"datasets/{prefix}_timestamps.npy", allow_pickle=True)


    #load train_indices and test_indices for specific scenario
    #contain indices for all of the k folds
    train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_{prefix}.json")


    #for task d - f:
    if param!=0:
        labels,timestamps=deduplicate_labels_and_timestamps(labels,timestamps, keep_inidces)
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_inidces)



    numeric_labels = encode_labels(global_label_encoder, labels) #make labels numeric
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack

    precision_all_folds=[]
    recall_all_folds=[]
    for fold_idx in range(k):
        precision, recall  = execute_fold_for_experiments(fold_idx, binary_numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],classifier=classifier, prefix=prefix, scenario=scenario,keep_indices=keep_inidces, param=param)
        precision_all_folds.append(precision)
        recall_all_folds.append(recall)

    return precision_all_folds, recall_all_folds



def execute_experiments_abc(global_label_encoder, k):
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