from sklearn.covariance import EllipticEnvelope
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score, f1_score
)
import numpy as np
import pandas as pd
import joblib


def elliptic_envelope_train(X_train, contamination=0.05):


    ee_clf = make_pipeline(
        StandardScaler(), #StandardScaler transforms each feature so that it has mean = 0 standard deviation = 1 -> required for ee!
        EllipticEnvelope(
            contamination=contamination, #expected fraction of anomalies in data
            support_fraction=None  # use full dataset for robust covariance estimation
        )
    )

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    ee_clf.fit(X_train)
    print("Elliptic Envelope training done")

    joblib.dump(ee_clf, "models/elliptic_envelope.joblib")

    return ee_clf




def elliptic_envelope_evaluate(X_test, y_test):

    try:
        ee_clf = joblib.load("models/elliptic_envelope.joblib")
    except FileNotFoundError:
        print("ERROR: elliptic_envelope model not found")
        return 0.0, 0.0, 0.0

    # binary labels: +1 normal, -1 attack
    y_test_binary = np.where(y_test == 0, 1, -1)

    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 1) Scores für ROC-AUC ---
    scores = ee_clf.decision_function(X_test)

    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"[EllipticEnvelope] ROC-AUC: {roc_auc:.4f}")

    # normal classification
    y_pred_numeric = ee_clf.predict(X_test)

    print("\n[EllipticEnvelope] --- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))

    # -1 = attack
    precision = precision_score(y_test_binary, y_pred_numeric, pos_label=-1)
    recall = recall_score(y_test_binary, y_pred_numeric, pos_label=-1)

    print(f"[EllipticEnvelope] Precision (Anomaly, -1): {precision:.4f}")
    print(f"[EllipticEnvelope] Recall    (Anomaly, -1): {recall:.4f}")

    return roc_auc, precision, recall


def elliptic_envelope_predict_error_overlap(X_test, y_test):
    """
    EE predict returns:
      +1 = inlier (Normal)
      -1 = outlier (Attack)
    Returns 1D error array (0/1) aligned to X_test/y_test order.
    """
    try:
        ee_clf = joblib.load("models/elliptic_envelope.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("models/elliptic_envelope.joblib not found. Please train first.")

    y_pred_numeric = ee_clf.predict(X_test)  # +1 normal, -1 attack
    y_pred_binary = np.where(y_pred_numeric == -1, 1, 0).astype(int)

    prediction_errors = (y_pred_binary != y_test).astype(int)
    return prediction_errors


def ee_f1_scorer(estimator, X, y_true):
    # EllipticEnvelope: +1=inlier(normal), -1=outlier(attack)
    y_pred = (estimator.predict(X) == -1).astype(int)  # -1 -> 1 (attack), +1 -> 0 (normal)
    return f1_score(y_true, y_pred, average="binary")


def ee_permutation_importance(
    X_test,
    y_test,
    n_samples=5000,
    n_repeats=3,
    random_state=0
):
    """
    Permutation feature importance for EllipticEnvelope using ROC-AUC.
    Uses a subsampled test set for efficiency.
    """
    ee_model= joblib.load("models/elliptic_envelope.joblib")

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

    # ---------------------------
    # Permutation importance
    # ---------------------------
    result = permutation_importance(
        ee_model,
        X_sub,
        y_sub,
        scoring=ee_f1_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    return result.importances_mean
