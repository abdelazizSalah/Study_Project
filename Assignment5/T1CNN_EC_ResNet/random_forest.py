from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import numpy as np
import pandas as pd

def binary_rf_train(X_train, y_train,
                    n_estimators=50,
                    max_depth=None,
                    random_state=42):


    rf_clf = RandomForestClassifier(
        n_estimators=n_estimators, #number of trees in the forest
        max_depth=max_depth,    #max depth of each tree (model complexity)
        random_state=random_state, ## ensures reproducible results by fixing randomness
        min_samples_split= 5,
        min_samples_leaf=1,
        n_jobs=-1
    )

    rf_clf.fit(X_train, y_train)
    joblib.dump(rf_clf, "models/brf.joblib")

    print("Random Forest training done")
    return rf_clf

def rf_permutation_importance(
    X_test,
    y_test,
    n_samples=3000,
    n_repeats=1,
    random_state=0,
    model_path="models/brf.joblib",
    scoring="f1"
):
    """
    Permutation feature importance for supervised binary Random Forest.
    Uses a subsampled test set for efficiency.
    Returns: result.importances_mean (can include small negatives).
    """
    rf_model = joblib.load(model_path)
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
        rf_model,
        X_sub,
        y_sub,
        scoring=scoring,          # keep consistent with KNN (you used "f1")
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    return result.importances_mean


def binary_rf_train_and_get_importance(X_train, y_train):

    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=0
    )
    rf.fit(X_train, y_train)

    # built-in RF importances (one value per input column)
    #measures each feature’s average total reduction in node impurity (e.g., Gini) from splits using that feature across all trees
    importances = rf.feature_importances_ #how much rf relied onneach feature to make a split decision and how much that split improved the trees

    joblib.dump(rf, "models/brf.joblib")

    return importances


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



def binary_rf_predict(X_test, t_test, return_binary_only: bool = False):
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

    if return_binary_only:
        return y_pred_numeric
    y_pred_attack = np.where(y_pred_numeric == 1, 'Attack', 'Normal')  # 1=attack, 0=no attack

    prediction_report = pd.DataFrame({
        'Timestamp (Unix)': t_test,
        'Predicted_Label': y_pred_attack
    })

    return prediction_report