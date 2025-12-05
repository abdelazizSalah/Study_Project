from Assignment3.Task2.labels_helper import encode_labels
import numpy as np

from Assignment3.Task2.svm import one_class_svm_train, one_class_svm_evaluate, split_training_and_test
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

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

#input: indices for one fold of one scenario!
def execute_fold_ee(ds, numeric_labels, timestamps,train_indices, test_indices):
    """train, measure result, print timestamp + prediction for test dataset for one fold"""
    X_train, X_test, y_train, y_test, t_train, t_test=split_training_and_test(ds, numeric_labels, timestamps,train_indices, test_indices)


    elliptic_envelope_train(X_train[:1000]) #todo debug - change (for faster model training)
    elliptic_envelope_evaluate(X_test, y_test)  #test data and corresponding labels
    prediction_report=elliptic_envelope_predict(X_test, t_test) #test data and corresponding timestamps
    print("\n--- Individual Packet Attack Detection Report ---")
    print(prediction_report)

    return




#train and test indices for each fold ([[][],...]
def execute_scenario_ee(ds, labels, timestamps, train_indices,test_indices, global_label_encoder):

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack
    #for fold_idx in range(len(test_indices)):
    #    execute_fold(ds, numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],ocsvm)

    first_fold_train_indices = train_indices[0]
    first_fold_test_indices = test_indices[0]
    print(len(train_indices[0]))
    execute_fold_ee(ds, binary_numeric_labels, timestamps, train_indices[0], test_indices[0])
    return