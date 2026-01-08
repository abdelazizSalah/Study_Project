from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import numpy as np


def binary_knn_train(
    X_train,
    y_train,
    n_neighbors=3,
    weights="uniform",     # "uniform" (each of the 5 neighbors contributes equally to the final decision)
    metric="manhattan",     # metric family to compute distance to neighbours
    p=2     #choose euclidean distance
):

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


    try:
        knn_clf = joblib.load("models/bknn.joblib")
    except FileNotFoundError:
        print("ERROR: 'models/bknn.joblib' not found. Please train the model first.")
        return 0.0, 0.0, 0.0, 0.0


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

    return precision, recall, f1



def binary_knn_predict_error_overlap(X_test, y_test):

    try:
        knn_clf = joblib.load("models/bknn.joblib")
    except FileNotFoundError:
        raise FileNotFoundError("models/bknn.joblib not found. Please train first.")

    y_pred_binary = knn_clf.predict(X_test).astype(int)
    prediction_errors = (y_pred_binary != y_test).astype(int)
    return prediction_errors


def knn_permutation_importance(
    X_test,
    y_test,
    n_samples=3000,
    n_repeats=1,
    random_state=0
):
    """
    Permutation feature importance for supervised binary KNN using ROC-AUC.
    Uses a subsampled test set for efficiency.
    """
    knn_model=joblib.load("models/bknn.joblib")
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
        knn_model,
        X_sub,
        y_sub,
        scoring="f1",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    return result.importances_mean