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


def normalize_pos_neg_to_unit(series):
    s = series.astype(float).copy()

    pos_max = s[s > 0].max() if (s > 0).any() else 1.0
    neg_max = abs(s[s < 0].min()) if (s < 0).any() else 1.0

    if pos_max == 0: pos_max = 1.0
    if neg_max == 0: neg_max = 1.0

    s.loc[s > 0] = s.loc[s > 0] / pos_max      # (0, 1]
    s.loc[s < 0] = s.loc[s < 0] / neg_max      # [-1, 0)

    return s


def plot_3_classifiers_sorted_by_helpfulness(
    prefix: str,
    scenario: int,
    classifiers: list[str],
    out_dir: str = "results/plots",
    top_n: int | None = None,      # None -> all features
    figsize_per_feature: float = 0.28,
    normalize: bool = True,        # <-- added
):
    assert len(classifiers) == 3, "Need exactly 3 classifiers."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # load series
    vecs = []
    for clf in classifiers:
        csv_path = f"results/feature_importance_{prefix}_s{scenario}_{clf}.csv"
        s = load_importance_csv(csv_path).astype(float)
        vecs.append(s)

    # align by common features (intersection)
    common = vecs[0].index
    for s in vecs[1:]:
        common = common.intersection(s.index)

    if len(common) == 0:
        raise ValueError(f"No common features across {classifiers} for {prefix} scenario {scenario}")

    vecs = [s.loc[common] for s in vecs]

    n_features = len(common)
    n_show = n_features if top_n is None else min(top_n, n_features)

    fig_h = max(4.0, figsize_per_feature * n_show)
    fig_w = 18.0
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)

    for ax, clf, s in zip(axes, classifiers, vecs):
        # normalize to [-1, 1] with separate scaling for pos/neg
        if normalize:
            s = normalize_pos_neg_to_unit(s)

        # sort by helpfulness: highest -> lowest (positives first, negatives last)
        s_sorted = s.sort_values(ascending=False).iloc[:n_show]

        feats_sorted = s_sorted.index.tolist()
        vals_sorted = s_sorted.values

        y = range(len(feats_sorted))
        ax.barh(y, vals_sorted)
        ax.set_yticks(list(y))
        ax.set_yticklabels(feats_sorted, fontsize=9)
        ax.invert_yaxis()

        ax.axvline(0, linewidth=1)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_title(clf)

        if normalize:
            ax.set_xlim(-1.0, 1.0)  # enforce shared comparable scale

    title = f"Feature helpfulness (sorted) — {prefix}, scenario {scenario} (top={n_show})"
    if normalize:
        title += " [normalized ± to 1]"
    fig.suptitle(title, fontsize=14)

    xlabel = "Importance score (positive = helpful, negative = harmful)"
    if normalize:
        xlabel += " — normalized per classifier (pos/max_pos, neg/|min_neg|)"
    fig.supxlabel(xlabel)

    out_path = f"{out_dir}/fi_helpful_panels_{prefix}_s{scenario}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path



def run_scenarios_for_feature_importance(
    k: int,
    global_label_encoder,
    prefix: str,                # "raw" or "re"
    keep_indices: int = 0,
    param: int = 0,
):
    scenario_models = {
        1: ["ocsvm", "lof", "ee"],
        2: ["rf", "knn", "bsvm"],
        3: ["rf", "knn", "bsvm"],
    }
    if prefix =="raw":
        scenario_models = { #todo remove
            2: ["rf"],
            3: ["rf", "bsvm"],
        }
    else:
        scenario_models = {  # todo remove
            2: ["rf", "bsvm"],
            3: ["rf", "bsvm"],
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
    return


def plot_all_feature_importance(out_dir="results/plots"):
    scenario_models = {
        1: ["ocsvm", "lof", "ee"],
        2: ["rf", "knn", "bsvm"],
        3: ["rf", "knn", "bsvm"],
    }

    for prefix in ["raw", "re15"]:
        for scenario, models in scenario_models.items():
            plot_3_classifiers_sorted_by_helpfulness(
                prefix=prefix,
                scenario=scenario,
                classifiers=models,
                out_dir=out_dir,
                top_n=None
            )