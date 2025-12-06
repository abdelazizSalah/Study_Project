from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import numpy as np
import pandas as pd

from labels_helper import encode_labels
from svm import split_training_and_test


def binary_rf_train(X_train, y_train,
                    n_estimators=100,
                    max_depth=None,
                    random_state=42):
    """
    Train a binary Random Forest classifier on (X_train, y_train).
    Labels: 0 = Control, 1 = Attack
    """

    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, "models/brf.joblib")

    print("Random Forest training done")
    return rf_clf



def binary_rf_evaluate(X_test, y_test):
    """
    Evaluate the Random Forest binary classifier.
    Uses ROC-AUC, precision, recall, F1 for the Attack class (1).
    """

    try:
        rf_clf = joblib.load("models/brf.joblib")
    except FileNotFoundError:
        print("FEHLER: 'models/brf.joblib' nicht gefunden. Bitte zuerst trainieren.")
        return 0.0, 0.0, 0.0, 0.0

    # --- 1) ROC-AUC using predicted probabilities for class 1 (Attack) ---
    if hasattr(rf_clf, "predict_proba"):
        scores = rf_clf.predict_proba(X_test)[:, 1]  # P(y=1|x)
    else:
        # Fallback: use decision_function or plain predictions
        # (RandomForestClassifier normally has predict_proba)
        scores = rf_clf.predict(X_test)

    roc_auc = roc_auc_score(y_test, scores)
    print(f"[RandomForest] ROC-AUC: {roc_auc:.4f}")

    # --- 2) Klassifikation ---
    y_pred_binary = rf_clf.predict(X_test)

    print("\n[RandomForest] --- Classification Report (0=Control, 1=Attack) ---")
    print(classification_report(y_test, y_pred_binary,
                                target_names=['Control (0)', 'Attack (1)']))

    precision = precision_score(y_test, y_pred_binary, pos_label=1)
    recall = recall_score(y_test, y_pred_binary, pos_label=1)
    f1 = f1_score(y_test, y_pred_binary, pos_label=1)

    print(f"[RandomForest] Precision (Attack, 1): {precision:.4f}")
    print(f"[RandomForest] Recall    (Attack, 1): {recall:.4f}")
    print(f"[RandomForest] F1 Score  (Attack, 1): {f1:.4f}")

    return roc_auc, precision, recall, f1


def binary_rf_predict(X_test, t_test):
    """
    For each measurement set at time t_test[i], predict Attack/Normal.
    Returns a DataFrame with timestamps and predicted labels.
    """

    try:
        rf_clf = joblib.load("models/brf.joblib")
    except FileNotFoundError:
        print("FEHLER: 'models/brf.joblib' nicht gefunden. Bitte zuerst trainieren.")
        return pd.DataFrame({
            'Timestamp (Unix)': [],
            'Predicted_Label': []
        })

    y_pred_numeric = rf_clf.predict(X_test)  # 0/1

    y_pred_attack = np.where(y_pred_numeric == 1, 'Attack', 'Normal')  # 1=attack, 0=no attack

    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_attack
    })

    return prediction_report


#input: indices for one fold of one scenario!
def execute_fold_rf(ds, numeric_labels, timestamps,train_indices, test_indices):
    """train, measure result, print timestamp + prediction for test dataset for one fold"""
    X_train, X_test, y_train, y_test, t_train, t_test=split_training_and_test(ds, numeric_labels, timestamps,train_indices, test_indices)
    #binary svm (multiple classes in training)
    binary_rf_train(X_train[:1000], y_train[:1000])
    binary_rf_evaluate(X_test, y_test)
    prediction_report=binary_rf_predict(X_test, t_test)
    print("\n--- Individual Packet Attack Detection Report ---")
    print(prediction_report)
    return


#train and test indices for each fold ([[][],...]
def execute_scenario_rf(ds, labels, timestamps, train_indices,test_indices, global_label_encoder):

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels=np.where(numeric_labels == 0, 0, 1) #convert from multiclass to binary labels 0 for control, 1 for attack
    #for fold_idx in range(len(test_indices)):
    #    execute_fold(ds, numeric_labels, timestamps, train_indices[fold_idx], test_indices[fold_idx],ocsvm)

    first_fold_train_indices = train_indices[0]
    first_fold_test_indices = test_indices[0]
    execute_fold_rf(ds, binary_numeric_labels, timestamps, train_indices[0], test_indices[0])
    return


