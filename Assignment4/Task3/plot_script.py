import matplotlib.pyplot as plt

# =========================
# DATA (best representative values)
# =========================
methods = [
    "GAN Raw - D",
    "GAN Raw - G",
    "GAN RE p=5 - D",
    "GAN RE p=5 - G",
    "GAN RE p=10 - D",
    "GAN RE p=10 - G",
    "GAN RE p=15 - D",
    "GAN RE p=15 - G",
    "N-grams Raw",
    "N-grams RE p=5",
    "N-grams RE p=10",
    "N-grams RE p=15",
]

precision = [
    0.9927,
    0.6264,
    0.4750,
    0.6923,
    0.6705,
    0.9512,
    1.0000,
    0.6667,
    0.9913,
    0.6356,
    0.6232,
    0.6628,
]

recall = [
    0.6241,
    0.0029,
    0.1348,
    0.0638,
    0.6373,
    0.3824,
    0.5281,
    0.1348,
    1.0000,
    1.0000,
    1.0000,
    0.9836,
]

f1 = [
    0.7664,
    0.0058,
    0.2099,
    0.1169,
    0.6211,
    0.5455,
    0.5949,
    0.2243,
    0.9956,
    0.7772,
    0.7679,
    0.7919,
]

# =========================
# PLOTTING FUNCTION
# =========================
def plot_metric(values, title, ylabel):
    plt.figure(figsize=(14, 6))
    plt.bar(methods, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

# =========================
# PLOTS
# =========================
plot_metric(f1, "F1 Score per Method", "F1 Score")
plot_metric(precision, "Precision per Method", "Precision")
plot_metric(recall, "Recall per Method", "Recall")

# Save plots as images .svg
def save_metric_plot(values, title, ylabel, filename):
    plt.figure(figsize=(14, 6))
    plt.bar(methods, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, format="svg")
    plt.close()
# Save the plots
save_metric_plot(f1, "F1 Score per Method", "F1 Score", "f1_score_plot.svg")
save_metric_plot(precision, "Precision per Method", "Precision", "precision_plot.svg")
save_metric_plot(recall, "Recall per Method", "Recall", "recall_plot.svg")

