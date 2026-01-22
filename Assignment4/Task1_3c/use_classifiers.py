###############################################################

import time
from local_outlier_factor import local_outlier_factor_evaluate, local_outlier_factor_train, \
    local_outlier_factor_predict_error_overlap, lof_permutation_importance
from labels_helper import deduplicate_labels_and_timestamps, deduplicate_folds
from knn import binary_knn_evaluate, binary_knn_train, binary_knn_predict_error_overlap, knn_permutation_importance
from elliptic_envelope import elliptic_envelope_train, elliptic_envelope_evaluate, \
    elliptic_envelope_predict_error_overlap, ee_permutation_importance
from file_helper_t3 import load_k_fold_results
from random_forest import binary_rf_train, binary_rf_evaluate, binary_rf_predict_error_overlap, \
    binary_rf_train_and_get_importance
from labels_helper import encode_labels
from svm import one_class_svm_evaluate, binary_svm_train, binary_svm_evaluate, one_class_svm_predict_error_overlap, \
    one_class_svm_train, binary_svm_predict_error_overlap, ocsvm_permutation_importance, get_bsvm_feature_importance
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


