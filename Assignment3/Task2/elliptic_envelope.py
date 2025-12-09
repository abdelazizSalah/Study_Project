from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def elliptic_envelope_train(X_train, contamination=0.05):
    """
    Train an Elliptic Envelope one-class model.
    contamination = expected fraction of anomalies in data
    """

    ee_clf = make_pipeline(
        StandardScaler(),
        EllipticEnvelope(
            contamination=contamination,
            support_fraction=None  # default robust covariance
        )
    )

    ee_clf.fit(X_train)
    print("Elliptic Envelope training done")

    joblib.dump(ee_clf, "models/elliptic_envelope.joblib")

    return ee_clf

from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score
)
import numpy as np
import pandas as pd
import joblib


def elliptic_envelope_evaluate(X_test, y_test):
    """
    Evaluate the Elliptic Envelope one-class model on a test set.

    y_test is expected to be multiclass/integers where
    0 = normal, everything else = attack.
    """

    try:
        ee_clf = joblib.load("models/elliptic_envelope.joblib")
    except FileNotFoundError:
        print("ERROR: elliptic_envelope model not found")
        return 0.0, 0.0, 0.0

    # binary labels: +1 normal, -1 attack
    y_test_binary = np.where(y_test == 0, 1, -1)

    # --- 1) Scores für ROC-AUC ---
    scores = ee_clf.decision_function(X_test)

    roc_auc = roc_auc_score(y_test_binary, scores)
    print(f"[EllipticEnvelope] ROC-AUC: {roc_auc:.4f}")

    # --- 2) Normale Klassifikation ---
    y_pred_numeric = ee_clf.predict(X_test)

    print(np.unique(y_test, return_counts=True))
    print(np.unique(y_test_binary, return_counts=True))
    print(np.unique(y_pred_numeric, return_counts=True))

    print("\n[EllipticEnvelope] --- Classification Report ---")
    print(classification_report(y_test_binary, y_pred_numeric))

    # Wir betrachten -1 als "Anomalie/Attacke"
    precision = precision_score(y_test_binary, y_pred_numeric, pos_label=-1)
    recall = recall_score(y_test_binary, y_pred_numeric, pos_label=-1)

    print(f"[EllipticEnvelope] Precision (Anomaly, -1): {precision:.4f}")
    print(f"[EllipticEnvelope] Recall    (Anomaly, -1): {recall:.4f}")

    return roc_auc, precision, recall


def elliptic_envelope_predict(X_test, t_test):
    """
    For each set of measurements at a given time step in the testing set,
    print/return whether this set of measurements contains any attack or not.

    Returns a DataFrame with:
      - 'Timestamp (Unix)'
      - 'Predicted_Label'  in {'Normal', 'Attack'}
    """

    try:
        ee_clf = joblib.load("models/elliptic_envelope.joblib")
    except FileNotFoundError:
        print("ERROR: elliptic_envelope model not found")
        return pd.DataFrame({
            'Timestamp (Unix)': [],
            'Predicted_Label': []
        })

    # +1 = Normal, -1 = Anomaly
    y_pred_numeric = ee_clf.predict(X_test)

    y_pred_attack = np.where(y_pred_numeric == -1, 'Attack', 'Normal')

    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_attack
    })

    return prediction_report
