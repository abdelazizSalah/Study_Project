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




