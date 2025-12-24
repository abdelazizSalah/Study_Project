from sklearn.inspection import permutation_importance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score, f1_score
)
import numpy as np
import joblib


def local_outlier_factor_train(X_train, contamination=0.05, n_neighbors=5):
    """
    Train a Local Outlier Factor (LOF) one-class model.
    contamination = expected fraction of anomalies in data
    n_neighbors   = number of neighbors used by LOF
    NOTE: We use novelty=True so that we can score *new* test samples.
    """

    lof_clf = make_pipeline(
        StandardScaler(),
        LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            leaf_size=30,
            novelty=True,        # IMPORTANT: enables predict/decision_function on unseen data
            n_jobs=-1            # use all cores
        )
    )

    lof_clf.fit(X_train)
    print("Local Outlier Factor training done")

    joblib.dump(lof_clf, "models/local_outlier_factor.joblib")

    return lof_clf


def local_outlier_factor_evaluate(X_test, y_test):

    try:
        lof_clf = joblib.load("models/local_outlier_factor.joblib")
    except FileNotFoundError:
        print("ERROR: local_outlier_factor model not found")
        return 0.0, 0.0, 0.0

    # binary labels: +1 normal, -1 attack
    y_test_binary = np.where(y_test == 0, 1, -1)

    # decision_function: larger values -> more "normal", smaller -> more "anomalous"
    scores = lof_clf.decision_function(X_test)

    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"[LocalOutlierFactor] ROC-AUC: {roc_auc:.4f}")

    # --- 2) Normale Klassifikation ---
    # LOF: +1 = inlier/normal, -1 = outlier/anomaly
    y_pred_numeric = lof_clf.predict(X_test)

    print("\n[LocalOutlierFactor] --- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))

    #-1 = attack
    precision = precision_score(y_test_binary, y_pred_numeric, pos_label=-1)
    recall = recall_score(y_test_binary, y_pred_numeric, pos_label=-1)

    print(f"[LocalOutlierFactor] Precision (Anomaly, -1): {precision:.4f}")
    print(f"[LocalOutlierFactor] Recall    (Anomaly, -1): {recall:.4f}")

    return roc_auc, precision, recall


def local_outlier_factor_predict_error_overlap(X_test, y_test):

    try:
        lof_clf = joblib.load("models/local_outlier_factor.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("local_outlier_factor model not found")

    # LOF convention:
    # +1 = inlier (Normal)
    # -1 = outlier (Attack)
    y_pred_numeric = lof_clf.predict(X_test)

    # convert BACK to binary labels: 1 = attack, 0 = normal
    y_pred_binary = np.where(y_pred_numeric == -1, 1, 0)

    # y_test is already binary: 1 = attack, 0 = normal
    prediction_errors = (y_pred_binary != y_test).astype(int)

    return prediction_errors

def lof_f1_scorer(estimator, X, y_true):
    # LOF: +1=inlier(normal), -1=outlier(attack)
    y_pred = (estimator.predict(X) == -1).astype(int)  # -1 -> attack(1)
    return f1_score(y_true, y_pred, average="binary")


def lof_permutation_importance(
    X_test,
    y_test,
    n_samples=3000,
    n_repeats=3,
    random_state=0
):
    """
    Permutation feature importance for Local Outlier Factor using ROC-AUC.
    Uses a subsampled test set for efficiency.
    """
    lof_model= joblib.load("models/local_outlier_factor.joblib")

    rng = np.random.default_rng(random_state)


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
        lof_model,
        X_sub,
        y_sub,
        scoring=lof_f1_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    return result.importances_mean