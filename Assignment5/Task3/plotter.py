import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# 1) Load your results file
# ----------------------------
FILE_PATH = "resnet_summary.csv"
df = pd.read_csv(FILE_PATH)
# ----------------------------
# 2) Basic cleaning / ordering
# ----------------------------
df["scenario"] = df["scenario"].astype(int)
df["use_stats"] = df["use_stats"].astype(int)

# Order representation like your screenshot
rep_order = ["raw", "re5", "re10", "re15"]
df["representation"] = pd.Categorical(df["representation"], categories=rep_order, ordered=True)

# Create a nice x label: "S2-raw", "S2-re5", ...
df["x_label"] = df.apply(lambda r: f"S{r['scenario']}-{r['representation']}", axis=1)

# Sort for stable plotting
df = df.sort_values(["scenario", "representation", "use_stats"]).reset_index(drop=True)

# ----------------------------
# 3) Plot helper (grouped bars by use_stats)
# ----------------------------
def plot_metric_grouped(metric_col: str, title: str, out_png: str):
    # Pivot to get two columns (use_stats 0 and 1) per x_label
    piv = df.pivot_table(
        index="x_label",
        columns="use_stats",
        values=metric_col,
        aggfunc="mean"
    ).fillna(0.0)

    # Make sure both columns exist (0 and 1), even if missing in data
    for c in [0, 1]:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv[[0, 1]]

    x = np.arange(len(piv.index))
    width = 0.38

    plt.figure(figsize=(max(10, len(piv.index) * 0.8), 5))
    plt.bar(x - width/2, piv[0].values, width=width, label="use_stats = 0")
    plt.bar(x + width/2, piv[1].values, width=width, label="use_stats = 1")

    plt.title(title)
    plt.ylabel(metric_col)
    plt.xticks(x, piv.index, rotation=45, ha="right")
    plt.ylim(0, 1.05)  # metrics are in [0,1]
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    # plt.show()

# ----------------------------
# 4) Create the 3 bar charts
# ----------------------------
os.makedirs("plots", exist_ok=True)

plot_metric_grouped("avg_precision", "Average Precision (grouped by use_stats)", "plots/avg_precision.svg")
plot_metric_grouped("avg_recall",    "Average Recall (grouped by use_stats)",    "plots/avg_recall.svg")
plot_metric_grouped("avg_f1",        "Average F1 (grouped by use_stats)",        "plots/avg_f1.svg")
