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

    pos_max = s[s > 0].max() if (s > 0).any() else 1.0  #finds largest positive
    neg_max = abs(s[s < 0].min()) if (s < 0).any() else 1.0 #finds largest negative

    if pos_max == 0: pos_max = 1.0
    if neg_max == 0: neg_max = 1.0

    s.loc[s > 0] = s.loc[s > 0] / pos_max      # (0, 1]
    s.loc[s < 0] = s.loc[s < 0] / neg_max      # [-1, 0)

    return s

#per scenario -datatype combination
def plot_3_classifiers_sorted_by_helpfulness(                       # plot 3 side-by-side feature-importance bar charts
    prefix: str,
    scenario: int,
    classifiers: list[str],
    out_dir: str = "results/plots",
    top_n: int | None = None,
    figsize_per_feature: float = 0.28,
    normalize: bool = True,
):
    assert len(classifiers) == 3, "Need exactly 3 classifiers."
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # load series                                                      # read feature-importance vectors from CSV files
    vecs = []
    for clf in classifiers:                                         # loop over each classifier name
        csv_path = f"results/feature_importance_{prefix}_s{scenario}_{clf}.csv"  # build the CSV path for this classifier
        s = load_importance_csv(csv_path).astype(float)             # load 1-row CSV into a Series and convert values to float
        vecs.append(s)                                              # store the Series

    n_features = len(vecs)                                        # total number of features available after alignment
    n_show = n_features if top_n is None else min(top_n, n_features) # decide how many features to actually display

    fig_h = max(4.0, figsize_per_feature * n_show)                  # figure height
    fig_w = 18.0                                                    # fixed figure width
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), constrained_layout=True)  # create a 1x3 subplot figure

    for ax, clf, s in zip(axes, classifiers, vecs):                 # iterate over axes, classifier names, and their Series together
        # normalize to [-1, 1] with separate scaling for pos/neg
        if normalize:
            s = normalize_pos_neg_to_unit(s)

        # sort by helpfulness: highest -> lowest (positives first, negatives last)
        s_sorted = s.sort_values(ascending=False)                   #sort descending

        feats_sorted = s_sorted.index.tolist()                      # list of feature names in sorted order
        vals_sorted = s_sorted.values                               # numpy array of corresponding importance values

        y = range(len(feats_sorted))                                # y positions for the horizontal bars (0..n_show-1)
        ax.barh(y, vals_sorted)                                     # draw horizontal bars for importance values
        ax.set_yticks(list(y))                                      # place y ticks at each bar position
        ax.set_yticklabels(feats_sorted, fontsize=9)                # label each bar with the feature name
        ax.invert_yaxis()                                           # put the highest-ranked feature at the top of the plot

        ax.axvline(0, linewidth=1)                                  # draw a vertical line at x=0 (separates + and -)
        ax.grid(True, axis="x", alpha=0.25)                         # add a light grid for the x-axis to help reading values
        ax.set_title(clf)                                           # title the panel with the classifier name

        if normalize:                                               # if normalization is enabled...
            ax.set_xlim(-1.0, 1.0)                                  # ...fix x-limits so all three panels share the same scale

    if prefix == "raw":                                             # choose a nicer title based on dataset type
        title = f"Feature Importance - RAW - Scenario {scenario}"    # title string for RAW
    else:
        title = f"Feature Importance - RE15 - Scenario {scenario}"   # title string for RE15 (or any non-raw prefix)
    fig.suptitle(title, fontsize=14)                                # set the overall (figure-level) title

    xlabel = "Importance score"                                     # x-axis label text for the whole figure
    fig.supxlabel(xlabel)                                           # set a shared x-axis label for all subplots

    out_path = f"{out_dir}/fi_helpful_panels_{prefix}_s{scenario}.png"  # output image path
    fig.savefig(out_path, dpi=200)                                  # save the figure as a PNG at 200 DPI
    plt.close(fig)                                                  # close the figure to free memory/resources
    return out_path                                                 # return the saved file path




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

    fn_prefix = f"re{param}" if prefix == "re" else "raw"
    Path("results").mkdir(exist_ok=True)

    feature_names = [f"f{i}" for i in range(32)]

    for scenario, models in scenario_models.items():
        scenario_importances = []

        for clf in models:
            imp = execute_scenario_feature_importance(
                global_label_encoder=global_label_encoder,
                classifier=clf,
                k=k,
                prefix=prefix,
                scenario=scenario,
                keep_indices=keep_indices,
                param=param
            )

            imp = np.asarray(imp).ravel() #collapses multi dimensional array into one dimensional
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