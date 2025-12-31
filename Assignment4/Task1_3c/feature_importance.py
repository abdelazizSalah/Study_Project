import csv
from pathlib import Path

from matplotlib import pyplot as plt

from use_classifiers import execute_scenario_feature_importance
from handling_re_bytes_integrated import get_keep_indices_from_fold0
import numpy as np
import pandas as pd

def load_importance_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    # one-row CSV: columns are feature names, first row is values
    return df.iloc[0]


def plot_compare_3_classifiers(
    prefix: str,              # "raw" or "re15" (matches your fn_prefix)
    scenario: int,            # 1, 2, 3
    classifiers: list[str],   # exactly 3 names in the order you want to show
    top_n: int = 12,
    out_dir: str = "results/plots"
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # load the 3 vectors
    vecs = []
    for clf in classifiers:
        csv_path = f"results/feature_importance_{prefix}_s{scenario}_{clf}.csv"
        s = load_importance_csv(csv_path).astype(float)
        vecs.append(s)

    # Align by columns (feature names)
    features = vecs[0].index.tolist()
    M = np.vstack([v.values for v in vecs])  # shape: (3, n_features)

    # pick top N features by mean importance across classifiers
    mean_imp = M.mean(axis=0)
    top_idx = np.argsort(mean_imp)[::-1][:top_n]

    top_features = [features[i] for i in top_idx]
    M_top = M[:, top_idx]  # shape: (3, top_n)

    # grouped bars
    x = np.arange(top_n)
    width = 0.25

    plt.figure()
    plt.bar(x - width, M_top[0], width, label=classifiers[0])
    plt.bar(x,         M_top[1], width, label=classifiers[1])
    plt.bar(x + width, M_top[2], width, label=classifiers[2])

    plt.xticks(x, top_features, rotation=45, ha="right")
    plt.ylabel("Feature importance")
    plt.title(f"Feature importance comparison — {prefix}, scenario {scenario}")
    plt.legend()
    plt.tight_layout()

    out_path = f"{out_dir}/fi_compare_{prefix}_s{scenario}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


def run_scenarios_for_feature_importance(
    k: int,
    global_label_encoder,
    prefix: str,                # "raw" or "re"
    keep_indices: int = 0,
    param: int = 0,
):
    scenario_models = {
        #1: ["ocsvm", "lof", "ee"], #todo change back!
        #2: ["rf", "knn", "bsvm"],
        3: ["rf", "knn", "bsvm"],
    }

    fn_prefix = f"re{param}" if prefix == "re" else "raw"
    Path("results").mkdir(exist_ok=True)

    # if you have real feature names, pass them in instead of f0..f31
    feature_names = [f"f{i}" for i in range(32)]

    for scenario, models in scenario_models.items():
        scenario_importances = []

        for clf in models:
            # IMPORTANT: this function should return a (32,) vector
            imp = execute_scenario_feature_importance(
                global_label_encoder=global_label_encoder,
                classifier=clf,
                k=k,
                prefix=prefix,
                scenario=scenario,
                keep_indices=keep_indices,
                param=param
            )

            imp = np.asarray(imp).ravel()
            if imp.shape[0] != 32:
                raise ValueError(f"{fn_prefix} s{scenario} {clf}: expected 32 features, got {imp.shape}")

            scenario_importances.append(imp)

            # write per-classifier importance
            csv_path = f"results/feature_importance_{fn_prefix}_s{scenario}_{clf}.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(feature_names)
                w.writerow(imp.tolist())
            print(f"[OK] wrote {csv_path}")

    return


def all_feature_importance(k: int, global_label_encoder):
    scenarios = [1, 2, 3]
    Path("results").mkdir(exist_ok=True)

    # ---------------- RAW ----------------
    run_scenarios_for_feature_importance(
        k=k,
        global_label_encoder=global_label_encoder,
        prefix="raw",
        keep_indices=0,
        param=0,
    )

    # ---------------- RE15 ----------------
    keep_indices = get_keep_indices_from_fold0("datasets/re_bytes_15", "re15")
    run_scenarios_for_feature_importance(
        k=k,
        global_label_encoder=global_label_encoder,
        prefix="re",
        keep_indices=keep_indices,
        param=15,
    )

    # ---------------- PLOTS ----------------
    scenario_models = {
        1: ["ocsvm", "lof", "ee"],
        2: ["rf", "knn", "bsvm"],
        3: ["rf", "knn", "bsvm"],
    }

    for prefix in ["raw", "re15"]:
        for scenario, models in scenario_models.items():
            plot_compare_3_classifiers(
                prefix=prefix,
                scenario=scenario,
                classifiers=models,
                top_n=32,  # plot all features
                out_dir="results/plots"
            )


    return

    return