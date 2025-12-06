from pyexpat import features

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import  classification_report, roc_auc_score, precision_score, recall_score, \
    f1_score
from labels_helper import  encode_labels
import pandas as pd


#########################################################OSVM

def one_class_svm_train(X_train):
    ocsvm_clf = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=0.05
        )
    )

    ocsvm_clf.fit(X_train)
    print("Model training done")
    joblib.dump(ocsvm_clf, "models/ocsvm.joblib")

    return ocsvm_clf


def one_class_svm_evaluate(X_test, y_test):

    """Make sure that for each classifier you identify and use the optimal grid
    parameters and compute the precision and recall for each of the experiments."""
    #todo: grid search

    try:
        ocsvm_clf = joblib.load("models/ocsvm.joblib")
    except FileNotFoundError:
        print("ERROR: oscmv model not found")
        return 0.0,0.0,0.0
    # binary labels: +1 normal, -1 attack
    y_test_binary = np.where(y_test == 0, 1, -1)

    # --- 1) Scores für ROC-AUC ---
    scores = ocsvm_clf.decision_function(X_test)

    # ROC-AUC berechnen
    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # --- 2) Normale Klassifikation ---
    y_pred_numeric = ocsvm_clf.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))

    precision = precision_score(y_test_binary, y_pred_numeric, pos_label=-1)    #tp/tp+fp
    recall = recall_score(y_test_binary, y_pred_numeric, pos_label=-1)  #tp/tp+fn

    print(f"Precision (Anomaly, -1): {precision:.4f}")
    print(f"Recall (Anomaly, -1): {recall:.4f}")
    return roc_auc, precision, recall


def one_class_svm_predict(X_test, t_test):
    """For each set of measurements at a given time step in the
    testing set, your piece of code should print out whether this set of measurements contains
    any attack or not."""
    #load

    try:
        ocsvm = joblib.load("models/ocsvm.joblib")
    except FileNotFoundError:
        print("ERROR: oscmv model not found")
        return pd.DataFrame({
            'Timestamp (Unix)': [],
            'Predicted_Label': []
        })

    # The predict method returns +1 for Normal and -1 for Anomaly
    y_pred_numeric = ocsvm.predict(X_test)

    y_pred_attack = np.where(y_pred_numeric == -1, 'Attack', 'Normal')  #-1=attack, 1=no attack
    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_attack
    })

    return prediction_report


##############################################################BSVM
#x - data, y - labels
def binary_svm_train(X_train, y_train):


    # SVC-Klassifikator mit linearem Kernel erstellen
    # C ist der Regularisierungsparameter. Kleinere Werte bedeuten stärkere Regularisierung.
    svm_clf = SVC(kernel='linear', C=1.0, random_state=42)

    # Modell trainieren (fitten)
    svm_clf.fit(X_train, y_train)
    joblib.dump(svm_clf, "models/bsvm.joblib")

    return svm_clf


def binary_svm_evaluate(X_test, y_test):
    try:
        svm_clf = joblib.load("models/bsvm.joblib")
    except FileNotFoundError:
        print("FEHLER: 'models/bsvm.joblib' nicht gefunden. Bitte zuerst trainieren.")
        return 0.0, 0.0, 0.0, 0.0

    scores = svm_clf.decision_function(X_test)

    # ROC-AUC berechnen (y_test_binary ist 0/1, Scores repräsentieren Klasse 1)
    roc_auc = roc_auc_score(y_test, scores)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # prediction
    y_pred_binary = svm_clf.predict(X_test)

    #evaluation
    print("\n--- Classification Report (0=Control, 1=Attack) ---")
    print(classification_report(y_test, y_pred_binary, target_names=['Control (0)', 'Attack (1)']))

    # attack=1
    precision = precision_score(y_test, y_pred_binary, pos_label=1)
    recall = recall_score(y_test, y_pred_binary, pos_label=1)
    f1 = f1_score(y_test, y_pred_binary, pos_label=1)

    print(f"Precision (Attack, 1): {precision:.4f}")
    print(f"Recall (Attack, 1): {recall:.4f}")
    print(f"F1 Score (Attack, 1): {f1:.4f}")

    return roc_auc, precision, recall, f1


def binary_svm_predict(X_test, t_test):
    try:
        svm_clf = joblib.load("models/bsvm.joblib")
    except FileNotFoundError:
        print("FEHLER: 'models/bsvm.joblib' nicht gefunden. Bitte zuerst trainieren.")
        return pd.DataFrame({
            'Timestamp (Unix)': [],
            'Predicted_Label': []
        })

    # The predict method returns +1 for Normal and -1 for Anomaly
    y_pred_numeric = svm_clf.predict(X_test)

    y_pred_attack = np.where(y_pred_numeric == 1, 'Attack', 'Normal')  # 1=attack, 0=no attack
    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_attack
    })

    return prediction_report


###############################################################

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
def execute_fold_svm(ds, numeric_labels, timestamps,train_indices, test_indices, ocsvm):
    """train, measure result, print timestamp + prediction for test dataset for one fold"""
    X_train, X_test, y_train, y_test, t_train, t_test=split_training_and_test(ds, numeric_labels, timestamps,train_indices, test_indices)


    if ocsvm:
        one_class_svm_train(X_train[:1000]) #todo debug - change (for faster model training)
        one_class_svm_evaluate(X_test, y_test)  #test data and corresponding labels
        prediction_report=one_class_svm_predict(X_test, t_test) #test data and corresponding timestamps
        print("\n--- Individual Packet Attack Detection Report ---")
        print(prediction_report)
    else:   #binary svm (multiple classes in training)
        binary_svm_train(X_train[:1000], y_train[:1000])
        binary_svm_evaluate(X_test, y_test)
        prediction_report=binary_svm_predict(X_test, t_test)
        print("\n--- Individual Packet Attack Detection Report ---")
        print(prediction_report)
    return


#train and test indices for each fold ([[][],...]
def execute_scenario_svm(ds, labels, timestamps, train_indices,test_indices, global_label_encoder, ocsvm):

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack
    #for fold_idx in range(len(test_indices)):
    #    execute_fold(ds, numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],ocsvm)

    first_fold_train_indices = train_indices[0]
    first_fold_test_indices = test_indices[0]
    execute_fold_svm(ds, binary_numeric_labels, timestamps, train_indices[0], test_indices[0],ocsvm)
    return


def train_and_save_models_for_al_scenarios():
    #todo preparation for task3

    pass