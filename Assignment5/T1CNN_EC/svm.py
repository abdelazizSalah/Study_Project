from pyexpat import features
from sys import prefix

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
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
            nu=0.01 # An upper bound on the fraction of training errors (outliers) and a lower bound on support vectors
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
    y_pred_binary = np.where(y_pred_numeric == -1, 1, 0).astype(int) #convert -1 to 1, 1 to 0

    prediction_errors = (y_pred_binary != y_test).astype(int) #prediction != label -> error
    return prediction_errors    #returns binary list -> 1 if error at that dp index, else 0


def ocsvm_f1_scorer(estimator, X, y_true):
    # OCSVM: +1=inlier(normal), -1=outlier(attack)
    y_pred = (estimator.predict(X) == -1).astype(int)   # -1 -> 1 (attack), +1 -> 0 (normal)
    return f1_score(y_true, y_pred, average="binary")


def ocsvm_permutation_importance(X_test, y_test, n_samples=2000, n_repeats=3,random_state=0):
    ocsvm_model= joblib.load("models/ocsvm.joblib")
    rng = np.random.default_rng(random_state)

    # ---------------------------
    # Subsample test data
    # ---------------------------
    if len(X_test) > n_samples:
        idx = rng.choice(len(X_test), size=n_samples, replace=False)
        X_sub = X_test[idx]
        y_sub = y_test[idx]
    else:
        X_sub = X_test
        y_sub = y_test


    #calculates base score,removes feature, calculates new score (improvement/harm)
    result = permutation_importance(
        ocsvm_model,
        X_sub,
        y_sub,
        scoring=ocsvm_f1_scorer,
        n_repeats=n_repeats, #overall it is repeated n times nd averaged
        random_state=random_state,
        n_jobs=-1
    )

    return result.importances_mean


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


    svm_clf = SVC(
        kernel='linear', # type of decision boundary; 'linear' fits a straight hyperplane
        gamma="scale",
        C=10,  # regularization strength; smaller C = more regularization (simpler model)
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


def get_bsvm_feature_importance():
    """
    importance : np.ndarray of shape (n_features,)
        Absolute value of the learned weight vector.
    """
    bsvm_model = joblib.load("models/bsvm.joblib")
    # If this is a pipeline, extract the final SVM step
    if hasattr(bsvm_model, "named_steps"):
        svm = bsvm_model[-1]
    else:
        svm = bsvm_model

    if not hasattr(svm, "coef_"):
        raise ValueError(
            "The provided SVM model does not expose coef_. "
            "Make sure kernel='linear' was used."
        )

    # coef_ shape: (1, n_features) for binary classification
    weights = svm.coef_.ravel()

    # importance = absolute contribution
    importance = np.abs(weights)

    return importance


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