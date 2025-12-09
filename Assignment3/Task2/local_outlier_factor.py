from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score
)
import numpy as np
import pandas as pd
import joblib


def local_outlier_factor_train(X_train, contamination=0.05, n_neighbors=20):
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
            novelty=True,        # IMPORTANT: enables predict/decision_function on unseen data
            n_jobs=-1            # use all cores
        )
    )

    lof_clf.fit(X_train)
    print("Local Outlier Factor training done")

    joblib.dump(lof_clf, "models/local_outlier_factor.joblib")

    return lof_clf


def local_outlier_factor_evaluate(X_test, y_test):
    """
    Evaluate the Local Outlier Factor one-class model on a test set.

    y_test is expected to be multiclass/integers where
    0 = normal, everything else = attack.
    """

    try:
        lof_clf = joblib.load("models/local_outlier_factor.joblib")
    except FileNotFoundError:
        print("ERROR: local_outlier_factor model not found")
        return 0.0, 0.0, 0.0

    # binary labels: +1 normal, -1 attack
    y_test_binary = np.where(y_test == 0, 1, -1)

    # --- 1) Scores für ROC-AUC ---
    # decision_function: larger values -> more "normal", smaller -> more "anomalous"
    scores = lof_clf.decision_function(X_test)

    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"[LocalOutlierFactor] ROC-AUC: {roc_auc:.4f}")

    # --- 2) Normale Klassifikation ---
    # LOF: +1 = inlier/normal, -1 = outlier/anomaly
    y_pred_numeric = lof_clf.predict(X_test)

    print("\n[LocalOutlierFactor] --- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))

    # Wir betrachten -1 als "Anomalie/Attacke"
    precision = precision_score(y_test_binary, y_pred_numeric, pos_label=-1)
    recall = recall_score(y_test_binary, y_pred_numeric, pos_label=-1)

    print(f"[LocalOutlierFactor] Precision (Anomaly, -1): {precision:.4f}")
    print(f"[LocalOutlierFactor] Recall    (Anomaly, -1): {recall:.4f}")

    return roc_auc, precision, recall


def local_outlier_factor_predict(X_test, t_test):
    """
    For each set of measurements at a given time step in the testing set,
    print/return whether this set of measurements contains any attack or not.

    Returns a DataFrame with:
      - 'Timestamp (Unix)'
      - 'Predicted_Label'  in {'Normal', 'Attack'}
    """

    try:
        lof_clf = joblib.load("models/local_outlier_factor.joblib")
    except FileNotFoundError:
        print("ERROR: local_outlier_factor model not found")
        return pd.DataFrame({
            'Timestamp (Unix)': [],
            'Predicted_Label': []
        })

    # +1 = Normal, -1 = Anomaly
    y_pred_numeric = lof_clf.predict(X_test)

    y_pred_attack = np.where(y_pred_numeric == -1, 'Attack', 'Normal')

    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_attack
    })

    return prediction_report
