from pyexpat import features
from sys import prefix

import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import  classification_report, roc_auc_score, precision_score, recall_score, \
    f1_score
import pandas as pd


#########################################################OSVM

def one_class_svm_train(X_train):
    ocsvm_clf = make_pipeline(
        StandardScaler(),
        OneClassSVM(
            kernel="rbf", ## Defines the function used to map data to a high-dimensional space; "rbf" = non-linear
            gamma="scale", ## Determines the influence of individual training samples; "scale" = self-adjusting calculation
            nu=0.05 # An upper bound on the fraction of training errors (outliers) and a lower bound on support vectors
        )
    )

    ocsvm_clf.fit(X_train)
    print("Model training done")
    joblib.dump(ocsvm_clf, "models/ocsvm.joblib")

    return ocsvm_clf


def one_class_svm_evaluate(X_test, y_test):

    try:
        ocsvm_clf = joblib.load("models/ocsvm.joblib")
    except FileNotFoundError:
        print("ERROR: oscmv model not found")
        return 0.0,0.0,0.0

    # convert binary labels: +1 normal, -1 attack
    y_test_binary = np.where(y_test == 0, 1, -1)

    # --- 1)ROC-AUC ---
    # may be better bc precision and recall depend on threshold and one class model provides anomaly score
    scores = ocsvm_clf.decision_function(X_test)

    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # --- 2) normal classification ---
    y_pred_numeric = ocsvm_clf.predict(X_test)

    print("\n--- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))   #prints a nice summary for precision, recall, f1

    precision = precision_score(y_test_binary, y_pred_numeric, pos_label=-1)    #tp/tp+fp
    recall = recall_score(y_test_binary, y_pred_numeric, pos_label=-1)  #tp/tp+fn

    print(f"Precision (Anomaly, -1): {precision:.4f}")
    print(f"Recall (Anomaly, -1): {recall:.4f}")
    return roc_auc, precision, recall


def one_class_svm_predict_error_overlap(X_test, y_test):

    try:
        ocsvm = joblib.load("models/ocsvm.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("models/ocsvm.joblib not found. Please train first.")

    y_pred_numeric = ocsvm.predict(X_test)  # +1 normal, -1 attack
    y_pred_binary = np.where(y_pred_numeric == -1, 1, 0).astype(int)

    prediction_errors = (y_pred_binary != y_test).astype(int)
    return prediction_errors


##############################################################BSVM
#x - data, y - labels
def binary_svm_train(X_train, y_train):


    svm_clf = SVC(
        kernel='linear', # type of decision boundary; 'linear' fits a straight hyperplane
        C=1.0,  # regularization strength; smaller C = more regularization (simpler model)
        random_state=42 # controls randomness in certain internal steps
    )

    # train model
    svm_clf.fit(X_train, y_train)
    joblib.dump(svm_clf, "models/bsvm.joblib")

    return svm_clf


def binary_svm_evaluate(X_test, y_test):
    try:
        svm_clf = joblib.load("models/bsvm.joblib")
    except FileNotFoundError:
        print("ERROR: ‘models/bsvm.joblib’ not found. Please train first.")
        return 0.0, 0.0, 0.0, 0.0

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

    return precision, recall, f1


def binary_svm_predict_error_overlap(X_test, y_test):

    try:
        svm_clf = joblib.load("models/bsvm.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("models/bsvm.joblib not found. Please train first.")

    y_pred_binary = svm_clf.predict(X_test).astype(int)

    prediction_errors = (y_pred_binary != y_test).astype(int)
    return prediction_errors




