###############################################################
import csv
import os
import time
import joblib
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score
from fine_tuning_optimized import grid_search_elliptic_envelope, grid_search_svm, grid_search_lof_parallel
from fine_tuning_optimized import grid_search_one_class_svm
from fine_tuning_optimized import grid_search_random_forest, grid_search_knn
from handling_re_bytes_integrated import get_keep_indices_from_fold0
from local_outlier_factor import local_outlier_factor_evaluate, local_outlier_factor_train, \
    local_outlier_factor_predict_error_overlap, lof_permutation_importance, local_outlier_factor_predict
from labels_helper import deduplicate_labels_and_timestamps, deduplicate_folds
from knn import binary_knn_evaluate, binary_knn_train, binary_knn_predict_error_overlap, knn_permutation_importance, \
    binary_knn_predict
from elliptic_envelope import elliptic_envelope_train, elliptic_envelope_evaluate, \
    elliptic_envelope_predict_error_overlap, ee_permutation_importance, elliptic_envelope_predict
from file_helper_t3 import load_k_fold_results
from random_forest import binary_rf_train, binary_rf_evaluate, binary_rf_predict_error_overlap, \
    binary_rf_train_and_get_importance, binary_rf_predict
from labels_helper import encode_labels
from svm import one_class_svm_evaluate, binary_svm_train, binary_svm_evaluate, one_class_svm_predict_error_overlap, \
    one_class_svm_train, binary_svm_predict_error_overlap, ocsvm_permutation_importance, get_bsvm_feature_importance, \
    one_class_svm_predict, binary_svm_predict
import numpy as np
import pandas as pd


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


def execute_fold_feature_importance(
    fold_idx, binary_numeric_labels, timestamps,
    train_indices, test_indices,
    classifier, prefix, scenario, param=0
):
    # ---- load features for this fold ----
    if param != 0:
        ds = np.load(f"datasets/re_bytes_{param}/re{param}_features_fold{fold_idx}.npy")
        print(f"Feature importance: {classifier}, fold {fold_idx}, scenario {scenario}, RE{param}")
    else:
        ds = np.load(f"datasets/{prefix}_features_fold{fold_idx}.npy")
        print(f"Feature importance: {classifier}, fold {fold_idx}, scenario {scenario}, RAW")

    X_train, X_test, y_train, y_test, _, _ = split_training_and_test(
        ds, binary_numeric_labels, timestamps, train_indices, test_indices
    )

    # ---- train (saves to disk) + load model ----
    if classifier == "ocsvm":
        one_class_svm_train(X_train)  # saves
        print("Calculating Importance Score OCSVM")
        importance_score = ocsvm_permutation_importance(
            X_test, y_test)

    elif classifier == "bsvm":
        binary_svm_train(X_train, y_train)
        print("Calculating Importance Score BSVM")
        importance_score = get_bsvm_feature_importance()

    elif classifier == "ee":
        elliptic_envelope_train(X_train)
        print("Calculating Importance Score EE")
        importance_score= ee_permutation_importance(
            X_test, y_test)

    elif classifier == "rf":
        print("Training and calculating Importance Score RF")
        importance_score = binary_rf_train_and_get_importance(X_train, y_train)

    elif classifier == "knn":
        binary_knn_train(X_train, y_train)
        print("Training and calculating Importance Score KNN")
        importance_score=knn_permutation_importance(
            X_test, y_test)

    elif classifier == "lof":
        local_outlier_factor_train(X_train)
        print("Training and calculating Importance Score LOF")
        importance_score = lof_permutation_importance(X_test, y_test)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    return importance_score #shape (32,)


def execute_scenario_feature_importance(
    global_label_encoder, classifier, k, prefix, scenario,
    keep_indices=0, param=0
):
    labels = np.load(f"datasets/{prefix}_labels.npy")
    timestamps = np.load(f"datasets/{prefix}_timestamps.npy", allow_pickle=True)

    train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_{prefix}.json")

    if param == 15:
        labels, timestamps = deduplicate_labels_and_timestamps(labels, timestamps, keep_indices)
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels = np.where(numeric_labels == 0, 0, 1)

    fold_importances = []

    for fold_idx in range(k):
        if len(train_indices[fold_idx]) == 0 or len(test_indices[fold_idx]) == 0:
            continue

        imp = execute_fold_feature_importance(
            fold_idx,
            binary_numeric_labels,
            timestamps,
            train_indices[fold_idx],
            test_indices[fold_idx],
            classifier=classifier,
            prefix=prefix,
            scenario=scenario,
            param=param
        )
        fold_importances.append(imp) #[[][][]] sublist per fold

    if len(fold_importances) == 0:
        return None

    return np.mean(np.vstack(fold_importances), axis=0) #stack sublists and average per column -> average per fold


#train and test indices for each fold ([[][],...]
def execute_scenario_rt(global_label_encoder, classifier, k, prefix, scenario, keep_indices=0, param=0):
    from measure_runtime import bytes_to_mb
    labels=np.load(f"datasets/{prefix}_labels.npy")
    timestamps = np.load(f"datasets/{prefix}_timestamps.npy", allow_pickle=True)


    #load train_indices and test_indices for specific scenario
    #contain indices for all of the k folds
    train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_{prefix}.json")


    #for re15:
    if param==15:
        labels,timestamps=deduplicate_labels_and_timestamps(labels,timestamps, keep_indices)
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)



    numeric_labels = encode_labels(global_label_encoder, labels) #make labels numeric
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack

    fold_runtimes_training=[]
    fold_runtimes_testing=[]
    fold_peak_rss_training=[]
    fold_peak_rss_testing=[]
    for fold_idx in range(k):

        if len(train_indices[fold_idx])==0 or len(test_indices[fold_idx])==0:
            continue

        if scenario in (2, 3):
            y_tr = binary_numeric_labels[train_indices[fold_idx]]
            if np.unique(y_tr).size < 2:
                continue

        runtime_training, peak_rss_training, runtime_testing, peak_rss_testing = execute_fold_rt(fold_idx, binary_numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],classifier=classifier, prefix=prefix, scenario=scenario,keep_indices=keep_indices, param=param)

        fold_runtimes_training.append(runtime_training)
        fold_runtimes_testing.append(runtime_testing)
        fold_peak_rss_training.append(peak_rss_training)
        fold_peak_rss_testing.append(peak_rss_testing)

    avg_runtime_training = np.mean(fold_runtimes_training)
    avg_peak_ram_training = bytes_to_mb(np.max(fold_peak_rss_training))

    avg_runtime_testing = np.mean(fold_runtimes_testing)
    avg_peak_ram_testing = bytes_to_mb(np.max(fold_peak_rss_testing))

    return avg_runtime_training,avg_peak_ram_training, avg_runtime_testing, avg_peak_ram_testing





def execute_fold_rt(fold_idx, binary_numeric_labels, timestamps,
                                 train_indices, test_indices, classifier, prefix, scenario, keep_indices=0, param=0):

    """Train with grid search, save best model, then use *_evaluate for metrics."""
    from measure_runtime import start_ram_monitor, stop_ram_monitor, bytes_to_mb

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
        # ---- TRAIN ----
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        one_class_svm_train(X_train)

        runtime_training = time.perf_counter() - start_time
        peak_rss_training = stop_ram_monitor(ram_handle)

        # ---- TEST ----
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        one_class_svm_evaluate(X_test, y_test)

        runtime_testing = time.perf_counter() - start_time
        peak_rss_testing = stop_ram_monitor(ram_handle)

    elif classifier == "bsvm":
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        binary_svm_train(X_train, y_train)

        runtime_training = time.perf_counter() - start_time
        peak_rss_training = stop_ram_monitor(ram_handle)

        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        binary_svm_evaluate(X_test, y_test)

        runtime_testing = time.perf_counter() - start_time
        peak_rss_testing = stop_ram_monitor(ram_handle)

    elif classifier == "ee":
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        elliptic_envelope_train(X_train)

        runtime_training = time.perf_counter() - start_time
        peak_rss_training = stop_ram_monitor(ram_handle)

        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        elliptic_envelope_evaluate(X_test, y_test)

        runtime_testing = time.perf_counter() - start_time
        peak_rss_testing = stop_ram_monitor(ram_handle)

    elif classifier == "rf":
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        binary_rf_train(X_train, y_train)

        runtime_training = time.perf_counter() - start_time
        peak_rss_training = stop_ram_monitor(ram_handle)

        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        binary_rf_evaluate(X_test, y_test)

        runtime_testing = time.perf_counter() - start_time
        peak_rss_testing = stop_ram_monitor(ram_handle)

    elif classifier == "knn":
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        binary_knn_train(X_train, y_train)

        runtime_training = time.perf_counter() - start_time
        peak_rss_training = stop_ram_monitor(ram_handle)

        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        binary_knn_evaluate(X_test, y_test)

        runtime_testing = time.perf_counter() - start_time
        peak_rss_testing = stop_ram_monitor(ram_handle)

    elif classifier == "lof":
        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        local_outlier_factor_train(X_train)

        runtime_training = time.perf_counter() - start_time
        peak_rss_training = stop_ram_monitor(ram_handle)

        ram_handle = start_ram_monitor(interval=0.1)
        start_time = time.perf_counter()

        local_outlier_factor_evaluate(X_test, y_test)

        runtime_testing = time.perf_counter() - start_time
        peak_rss_testing = stop_ram_monitor(ram_handle)

    else:
        print("wrong classifier choice")
        return 0.0, 0.0, 0.0, 0.0

    return (
        runtime_training, peak_rss_training, runtime_testing, peak_rss_testing,
    )


def execute_fold_error_overlap(fold_idx, binary_numeric_labels, timestamps,
                                 train_indices, test_indices, classifier, prefix, scenario, keep_indices=0, param=0):


    if param!=0:    #re15
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
        one_class_svm_train(X_train)
        prediction_errors=one_class_svm_predict_error_overlap(X_test, y_test) #[0,1,1,0,...] prediction errors for fold

    elif classifier == "bsvm":

        binary_svm_train(X_train, y_train)

        prediction_errors=binary_svm_predict_error_overlap(X_test, y_test)

    elif classifier == "ee":
        elliptic_envelope_train(X_train)

        prediction_errors=elliptic_envelope_predict_error_overlap(X_test, y_test)

    elif classifier == "rf":
        binary_rf_train(X_train, y_train)

        prediction_errors=binary_rf_predict_error_overlap(X_test, y_test)

    elif classifier == "knn":
        binary_knn_train(X_train, y_train)

        prediction_errors=binary_knn_predict_error_overlap(X_test, y_test)

    elif classifier == "lof":
        local_outlier_factor_train(X_train)
        prediction_errors=local_outlier_factor_predict_error_overlap(X_test, y_test)

    else:
        print("wrong classifier choice")
        return 0.0, 0.0, 0.0, 0.0

    return prediction_errors #returns list [0,1,0,1,1,0,0,...] for testing for fold -> 1 for error, 0 for no error


#executes scenario for one classifier!!
def execute_scenario_error_overlap(global_label_encoder, classifier, k, prefix, scenario, keep_indices=0, param=0):
    labels=np.load(f"datasets/{prefix}_labels.npy")
    timestamps = np.load(f"datasets/{prefix}_timestamps.npy", allow_pickle=True)


    #load train_indices and test_indices for specific scenario
    #contain indices for all of the k folds
    train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_{prefix}.json")


    #for re15:
    if param==15:
        labels,timestamps=deduplicate_labels_and_timestamps(labels,timestamps, keep_indices)
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices) #[[32432, 23, 334, ...][][][]]


    numeric_labels = encode_labels(global_label_encoder, labels) #make labels numeric
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack

    prediction_errors_all_folds=[]
    for fold_idx in range(k):

        prediction_errors_fold = execute_fold_error_overlap(fold_idx, binary_numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],classifier=classifier, prefix=prefix, scenario=scenario,keep_indices=keep_indices, param=param)

        #summarize prediction_errors for all folds for whole dataset
        #first iteration eg results for test indices [1,2,3,4,..]

        #second iteration for [5,6,7,8,...]
        prediction_errors_all_folds.extend(prediction_errors_fold.tolist())


    return prediction_errors_all_folds ##returns list [0,1,0,1,1,0,0,...] for testing for all folds -> 1 for error, 0 for no error


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


##########################################################################################################

def execute_fold_for_ensemble_classifier(method,multiclass, fold_idx, binary_numeric_labels, timestamps,
                                 train_indices, test_indices, prefix, scenario, param=0):
    """Train with grid search, save best model, then use *_evaluate for metrics."""

    from ensemble_classifier import get_ec_prediction

    if param!=0:
        ds = np.load(f"datasets/re_bytes_{param}/re{param}_features_fold{fold_idx}.npy")
        print(f"Executing {multiclass} for fold {fold_idx} in Scenario {scenario} for RE{param}.")
    else:
        ds = np.load(f"datasets/{prefix}_features_fold{fold_idx}.npy")
        print(f"Executing {multiclass} for fold {fold_idx} in Scenario {scenario} for RAW.")
    #for task d - f


    X_train, X_test, y_train, y_test, t_train, t_test = split_training_and_test(
        ds, binary_numeric_labels, timestamps, train_indices, test_indices
    )


    # run on small portion of dataset, for demonstration:
    first_indices = np.arange(0, 1000)
    last_indices = np.arange(len(X_train) - 1000, len(X_train))
    selected_indices = np.concatenate((first_indices, last_indices))
    X_train = X_train[selected_indices]
    y_train = y_train[selected_indices]

    if not multiclass:
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Training One-Class SVM...")
        one_class_svm_train(X_train)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Predicting One-Class SVM...")
        p1 = one_class_svm_predict(X_test, t_test, return_binary_only=True)

        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Training Elliptic Envelope...")
        elliptic_envelope_train(X_train)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Predicting Elliptic Envelope...")
        p2 = elliptic_envelope_predict(X_test, t_test, return_binary_only=True)

        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Training LOF...")
        local_outlier_factor_train(X_train)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Predicting LOF...")
        p3 = local_outlier_factor_predict(X_test, t_test, return_binary_only=True)

    else:
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Training Binary RF (model 1)...")
        binary_rf_train(X_train, y_train)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Predicting Binary RF (model 1)...")
        p1 = binary_rf_predict(X_test, t_test, return_binary_only=True)

        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Training Binary KNN (model 2)...")
        binary_knn_train(X_train, y_train)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Predicting Binary KNN (model 2)...")
        p2 = binary_knn_predict(X_test, t_test, return_binary_only=True)

        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Training Binary RF (model 3)...")
        binary_rf_train(X_train, y_train)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Evaluating Binary RF (model 3)...")
        binary_rf_evaluate(X_test, y_test)
        print(f"[fold {fold_idx} | scen {scenario} | {prefix}] Predicting Binary RF (model 3)...")
        p3 = binary_rf_predict(X_test, t_test, return_binary_only=True)


    #if method is all (for experiments)
    methods = ["random", "majority", "all"] if method == "all_methods" else [method]

    precision = {}
    recall = {}
    f1={}
    for m in methods:   #dictinary
        p = get_ec_prediction(p1, p2, p3, m)
        precision[m] = float(precision_score(y_test, p, zero_division=0))
        recall[m] = float(recall_score(y_test, p, zero_division=0))
        f1[m] = float(f1_score(y_test, p, zero_division=0))

    return precision, recall, f1


#train and test indices for each fold ([[][],...]
def execute_scenario_for_ensemble_classifier(method,multiclass,global_label_encoder, k, prefix, scenario, keep_indices=0, param=0):


    #for task d - f:
    if param!=0:
        # load train_indices and test_indices for specific scenario
        # contain indices for all of the k folds
        train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_re.json")
        labels = np.load(f"datasets/re_labels.npy")
        timestamps = np.load(f"datasets/re_timestamps.npy", allow_pickle=True)
        labels,timestamps=deduplicate_labels_and_timestamps(labels,timestamps, keep_indices)
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)
    else:
        train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s{scenario}_raw.json")
        labels = np.load(f"datasets/raw_labels.npy")
        timestamps = np.load(f"datasets/raw_timestamps.npy", allow_pickle=True)
    numeric_labels = encode_labels(global_label_encoder, labels) #make labels numeric
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack

    precision_all_folds=[]
    recall_all_folds=[]
    f1_all_folds = []

    for fold_idx in range(k):
        if len(train_indices[fold_idx])==0 or len(test_indices[fold_idx])==0:
            continue

        if scenario in (2, 3):
            y_tr = binary_numeric_labels[train_indices[fold_idx]]
            if np.unique(y_tr).size < 2:
                continue

        # dict which contains value for each method in case of "all_methods"
        prec_dict, rec_dict, f1_dict = execute_fold_for_ensemble_classifier(
            method, multiclass, fold_idx, binary_numeric_labels, timestamps,
            train_indices[fold_idx], test_indices[fold_idx],
            prefix=prefix, scenario=scenario, param=param
        )

        precision_all_folds.append(prec_dict)
        recall_all_folds.append(rec_dict)
        f1_all_folds.append(f1_dict)

    # compute mean over folds per method
    methods = list(precision_all_folds[0].keys())  # preserves insertion order

    avg_precision = {}
    avg_recall = {}
    avg_f1 = {}

    for m in methods:
        precisions_for_m = []
        recalls_for_m = []
        f1s_for_m = []

        for fold_dict in precision_all_folds:
            precisions_for_m.append(fold_dict[m])

        for fold_dict in recall_all_folds:
            recalls_for_m.append(fold_dict[m])

        for fold_dict in f1_all_folds:
            f1s_for_m.append(fold_dict[m])
        avg_precision[m] = float(sum(precisions_for_m) / len(precisions_for_m))
        avg_recall[m] = float(sum(recalls_for_m) / len(recalls_for_m))
        avg_f1[m] = float(sum(f1s_for_m) / len(f1s_for_m))

    # return lists in the same method order
    return (
        [avg_precision[m] for m in methods], #list [pr_method1,pr_method2,pr_method3]
        [avg_recall[m] for m in methods],   #[re_method1,re_method2,re_method3]
        [avg_f1[m] for m in methods],
    )
######################################################################


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
        joblib.dump(best_model, "models/ocsvm.joblib")
        roc_auc, precision, recall = one_class_svm_evaluate(X_test, y_test)

    elif classifier == "bsvm":
        base = SVC(probability=True, random_state=42)
        best_model = grid_search_svm(base, X_train, y_train, X_test, y_test, scoring_metric="f1_macro")
        joblib.dump(best_model, "models/bsvm.joblib")  # adjust name if different in your code
        precision, recall, f1 = binary_svm_evaluate(X_test, y_test)

    elif classifier == "ee":
        base = EllipticEnvelope()
        best_model = grid_search_elliptic_envelope(base, X_train, y_train, X_test, y_test, scoring_metric="accuracy")
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