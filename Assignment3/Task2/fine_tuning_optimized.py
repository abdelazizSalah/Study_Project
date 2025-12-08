N_JOBLIB = 64

from itertools import product

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import f1_score, accuracy_score, recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import KNeighborsClassifier


# -------------------------------------------------------------------
# Helper functions (shared by all grid searches)
# -------------------------------------------------------------------

def _iter_param_grid(param_grid: dict):
    """Yield all combinations of a param grid as dicts."""
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def _get_supervised_scorer(scoring_metric: str):
    """
    Return a scorer function(y_true, y_pred) for common metrics.
    Used for supervised models (RF / SVM / KNN).
    """
    if scoring_metric in ("f1", "f1_binary"):
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average="binary")
    if scoring_metric == "f1_macro":
        return lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
    if scoring_metric == "accuracy":
        return accuracy_score
    if scoring_metric == "recall":
        return recall_score
    raise ValueError(f"Unsupported scoring_metric: {scoring_metric}")


def _eval_params_supervised(base_model, params, X_train, y_train, X_val, y_val, scorer):
    """
    Fit + score ONE parameter combination for supervised models.
    Runs in parallel via joblib.Parallel.
    Returns (score, params, fitted_model).
    """
    model = clone(base_model)
    model.set_params(**params)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    score = scorer(y_val, y_val_pred)
    return score, params, model


# -------------------------------------------------------------------
# Random Forest
# -------------------------------------------------------------------

def grid_search_random_forest(
    rf: RandomForestClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
    scoring_metric="f1_macro",
    n_jobs=None,
):
    """
    Fast & parallel Random Forest grid search using a single train/val split.
    Signature is compatible with your old code.
    """
    print("Starting FAST & PARALLEL Random Forest Grid Search...")

    if n_jobs is None:
        n_jobs = N_JOBLIB

    # You can shrink/enlarge this grid if needed
    # param_grid = {
    #     "n_estimators": [100, 200],
    #     "max_depth": [None, 10, 20],
    #     "min_samples_split": [2, 5],
    #     "min_samples_leaf": [1, 2],
    # }
    param_grid = {
        "n_estimators": 50,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
    }

    scorer = _get_supervised_scorer(scoring_metric)
    all_params = list(_iter_param_grid(param_grid))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_params_supervised)(
            rf, params, X_train, y_train, X_val, y_val, scorer
        )
        for params in all_params
    )

    best_score, best_params, best_model = max(results, key=lambda tpl: tpl[0])

    print(f"Best parameters: {best_params}")
    print(f"Validation {scoring_metric}: {best_score:.4f}")

    return best_model


# -------------------------------------------------------------------
# SVM
# -------------------------------------------------------------------

def grid_search_svm( # 0.932, 0.937 = 1, 
    svm: SVC,
    X_train,
    y_train,
    X_val,
    y_val,
    scoring_metric="f1_macro",
    n_jobs=None,
):
    """
    Fast & parallel SVM grid search.
    """
    print("Starting FAST & PARALLEL SVM Grid Search...")

    if n_jobs is None:
        n_jobs = N_JOBLIB

    param_grid = {
        "C": 10,
        "kernel": 'linear',
        "gamma": 'scale',
    }

    scorer = _get_supervised_scorer(scoring_metric)
    all_params = list(_iter_param_grid(param_grid))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_params_supervised)(
            svm, params, X_train, y_train, X_val, y_val, scorer
        )
        for params in all_params
    )

    best_score, best_params, best_model = max(results, key=lambda tpl: tpl[0])

    print(f"Best parameters: {best_params}")
    print(f"Validation {scoring_metric}: {best_score:.4f}")

    return best_model


# -------------------------------------------------------------------
# KNN
# -------------------------------------------------------------------

def grid_search_knn( #1.  9820,  
    knn: KNeighborsClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
    scoring_metric="f1_macro",
    n_jobs=None,
):
    """
    Fast & parallel KNN grid search.
    """
    print("Starting FAST & PARALLEL KNN Grid Search...")

    if n_jobs is None:
        n_jobs = N_JOBLIB

    param_grid = {
        "n_neighbors": 3,
        "weights": "uniform",
        "metric": "manhattan",
    }

    scorer = _get_supervised_scorer(scoring_metric)
    all_params = list(_iter_param_grid(param_grid))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_params_supervised)(
            knn, params, X_train, y_train, X_val, y_val, scorer
        )
        for params in all_params
    )

    best_score, best_params, best_model = max(results, key=lambda tpl: tpl[0])

    print(f"Best parameters: {best_params}")
    print(f"Validation {scoring_metric}: {best_score:.4f}")

    return best_model


# -------------------------------------------------------------------
# One-Class SVM (anomaly detection)
# -------------------------------------------------------------------

def _eval_params_ocsvm(base_model, params, X_train, X_val, y_val): # 
    """
    Fit + score ONE parameter combination for One-Class SVM.
    Uses accuracy on the anomaly/normal mapping.
    """
    model = clone(base_model)
    model.set_params(**params)
    model.fit(X_train)  # unsupervised

    y_val_pred_raw = model.predict(X_val)          # -1 anomaly, 1 normal
    y_val_pred = (y_val_pred_raw == -1).astype(int)  # 1 = anomaly/attack
    score = accuracy_score(y_val, y_val_pred)
    return score, params, model


def grid_search_one_class_svm(
    ocsvm: OneClassSVM,
    X_train,
    y_train,   # not used, kept for API compatibility
    X_val,
    y_val,
    scoring_metric="accuracy",
    n_jobs=None,
):
    """
    Fast & parallel grid search for One-Class SVM.
    Optimises accuracy on the validation set.
    """
    print("Starting FAST & PARALLEL One-Class SVM Grid Search...")

    if n_jobs is None:
        n_jobs = N_JOBLIB

    param_grid = { # .8,.45 -- 
        "kernel": 'rbf',
        "gamma": 'scale',
        "nu": 0.01,
    }

    all_params = list(_iter_param_grid(param_grid))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_params_ocsvm)(
            ocsvm, params, X_train, X_val, y_val
        )
        for params in all_params
    )

    best_score, best_params, best_model = max(results, key=lambda tpl: tpl[0])

    print(f"Best parameters: {best_params}")
    print(f"Validation Accuracy: {best_score:.4f}")

    return best_model


# -------------------------------------------------------------------
# Elliptic Envelope (anomaly detection)
# -------------------------------------------------------------------

def _ee_scorer(y_true, y_pred, scoring_metric):
    if scoring_metric.startswith("f1"):
        return f1_score(y_true, y_pred, average="binary")
    if scoring_metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if scoring_metric == "recall":
        return recall_score(y_true, y_pred)
    raise ValueError(f"Unsupported scoring_metric for EE: {scoring_metric}")


def _eval_params_ee(base_model, params, X_train, X_val, y_val, scoring_metric):
    """
    Fit + score ONE parameter combination for EllipticEnvelope.
    """
    model = clone(base_model)
    model.set_params(**params)
    model.fit(X_train)

    y_val_pred_raw = model.predict(X_val)          # -1 anomaly, 1 normal
    y_val_pred = (y_val_pred_raw == -1).astype(int)
    score = _ee_scorer(y_val, y_val_pred, scoring_metric)
    return score, params, model


def grid_search_elliptic_envelope(
    ee: EllipticEnvelope,
    X_train,
    y_train,    # not used, kept for API compatibility
    X_val,
    y_val,
    scoring_metric="f1",
    n_jobs=None,
):
    """
    Fast & parallel EllipticEnvelope grid search.
    """
    print("Starting FAST & PARALLEL Elliptic Envelope Grid Search...")

    if n_jobs is None:
        n_jobs = N_JOBLIB

    param_grid = {
        "contamination": 0.05,
        "support_fraction": None,
    }

    all_params = list(_iter_param_grid(param_grid))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_eval_params_ee)(
            ee, params, X_train, X_val, y_val, scoring_metric
        )
        for params in all_params
    )

    best_score, best_params, best_model = max(results, key=lambda tpl: tpl[0])

    print(f"Best parameters: {best_params}")
    print(f"Validation {scoring_metric}: {best_score:.4f}")

    return best_model