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


