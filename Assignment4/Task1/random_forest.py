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


    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators, #number of trees in the forest
        max_depth=max_depth,    #max depth of each tree (model complexity)
        random_state=random_state, ## ensures reproducible results by fixing randomness
        n_jobs=-1
    )

    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, "models/brf.joblib")

    print("Random Forest training done")
    return rf_clf



def binary_rf_evaluate(X_test, y_test):

    try:
        rf_clf = joblib.load("models/brf.joblib")
    except FileNotFoundError:
        print("ERROR: ‘models/bsvm.joblib’ not found. Please train first.")
        return 0.0, 0.0, 0.0, 0.0


    # --- classification ---
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

    return precision, recall, f1


def binary_rf_predict_error_overlap(X_test, y_test):

    try:
        rf_clf = joblib.load("models/brf.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("models/brf.joblib not found. Please train first.")

    y_pred_binary = rf_clf.predict(X_test).astype(int)

    prediction_errors = (y_pred_binary != y_test).astype(int)
    return prediction_errors


