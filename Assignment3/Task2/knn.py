from sklearn.neighbors import KNeighborsClassifier
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


def binary_knn_train(
    X_train,
    y_train,
    n_neighbors=5,
    weights="uniform",     # options: "uniform" or "distance"
    metric="minkowski",     # default Minkowski (p=2 = Euclidean)
    p=2
):
    """
    Train a binary k-Nearest Neighbors classifier on (X_train, y_train).
    Labels: 0 = Control, 1 = Attack
    """

    knn_clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        p=p
    )

    knn_clf.fit(X_train, y_train)
    joblib.dump(knn_clf, "models/bknn.joblib")

    print("kNN training completed.")
    return knn_clf



def binary_knn_evaluate(X_test, y_test):
    """
    Evaluate the kNN binary classifier.
    Metrics: ROC-AUC, precision, recall, F1 for the Attack class (label=1).
    """

    try:
        knn_clf = joblib.load("models/bknn.joblib")
    except FileNotFoundError:
        print("ERROR: 'models/bknn.joblib' not found. Please train the model first.")
        return 0.0, 0.0, 0.0, 0.0

    # --- ROC-AUC using predicted probabilities for class 1 ---
    if hasattr(knn_clf, "predict_proba"):
        scores = knn_clf.predict_proba(X_test)[:, 1]
    else:
        scores = knn_clf.predict(X_test)

    roc_auc = roc_auc_score(y_test, scores)
    print(f"[kNN] ROC-AUC: {roc_auc:.4f}")

    # --- Classification Report ---
    y_pred_binary = knn_clf.predict(X_test)

    print("\n[kNN] --- Classification Report (0=Control, 1=Attack) ---")
    print(classification_report(
        y_test,
        y_pred_binary,
        target_names=['Control (0)', 'Attack (1)']
    ))

    precision = precision_score(y_test, y_pred_binary, pos_label=1)
    recall = recall_score(y_test, y_pred_binary, pos_label=1)
    f1 = f1_score(y_test, y_pred_binary, pos_label=1)

    print(f"[kNN] Precision (Attack=1): {precision:.4f}")
    print(f"[kNN] Recall    (Attack=1): {recall:.4f}")
    print(f"[kNN] F1 Score  (Attack=1): {f1:.4f}")

    return roc_auc, precision, recall, f1



def binary_knn_predict(X_test, t_test):
    """
    Predict Attack/Normal labels for each sample in X_test.
    Returns a DataFrame with timestamps and predicted labels.
    """

    try:
        knn_clf = joblib.load("models/bknn.joblib")
    except FileNotFoundError:
        print("ERROR: 'models/bknn.joblib' not found. Please train the model first.")
        return pd.DataFrame({
            'Timestamp (Unix)': [],
            'Predicted_Label': []
        })

    # Numeric predictions: 0 = Control, 1 = Attack
    y_pred_numeric = knn_clf.predict(X_test)

    # Convert to textual labels
    y_pred_text = np.where(y_pred_numeric == 1, "Attack", "Normal")

    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_text
    })

    return prediction_report
