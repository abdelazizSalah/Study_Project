###############################################################
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

    # FEATURES AUFTEILEN
    X_train = ds[train_indices_fold]
    X_test = ds[test_indices_fold]

    # LABELS AUFTEILEN
    y_train = labels[train_indices_fold]
    y_test = labels[test_indices_fold]

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


    if classifier=="ocsvm":
        one_class_svm_train(X_train) #todo debug - change (for faster model training)
        one_class_svm_evaluate(X_test, y_test)  #test data and corresponding labels
        prediction_report=one_class_svm_predict(X_test, t_test) #test data and corresponding timestamps
    elif classifier=="bsvm":   #binary svm (multiple classes in training)
        binary_svm_train(X_train, y_train)
        binary_svm_evaluate(X_test, y_test)
        prediction_report=binary_svm_predict(X_test, t_test)
    elif classifier == "ee":
        elliptic_envelope_train(X_train)  # todo debug - change (for faster model training)
        elliptic_envelope_evaluate(X_test, y_test)  # test data and corresponding labels
        prediction_report = elliptic_envelope_predict(X_test, t_test)  # test data and corresponding timestamps

    elif classifier == "rf":
        binary_rf_train(X_train, y_train) # todo debug - change (for faster model training)
        binary_rf_evaluate(X_test, y_test)
        prediction_report = binary_rf_predict(X_test, t_test)
    elif classifier == "knn":
        binary_knn_train(X_train, y_train) # todo debug - change (for faster model training)
        binary_knn_evaluate(X_test, y_test)
        prediction_report = binary_knn_predict(X_test, t_test)
    else:
        print("wrong classifier choice")

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