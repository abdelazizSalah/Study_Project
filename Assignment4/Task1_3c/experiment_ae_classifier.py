import csv
from pathlib import Path

from matplotlib import pyplot as plt
from tensorflow import keras

from feature_creation_autoencoder import train_and_save_models_classifier
from handling_re_bytes_integrated import  get_keep_indices_from_fold0_ae
from labels_helper import deduplicate_folds
from labels_helper import deduplicate_labels_and_timestamps, encode_labels
from file_helper_t3 import load_k_fold_results
import numpy as np


####plots
def read_ae_summary(csv_path="results/ae_summary.csv"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rep = row["representation"].strip()

            data[rep] = (
                float(row["avg_precision"]),
                float(row["avg_recall"]),
                float(row["avg_f1"]),
            )
    return data

def save_barplot(order, values, title, ylabel, out_path):
    fig, ax = plt.subplots()

    bars = ax.bar(order, values)
    ax.set_ylim(0.0, 1.08)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Dataset Type")
    ax.set_ylabel(ylabel)

    # value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def make_ae_metric_plots(summary_csv="results/ae_summary.csv", out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_ae_summary(summary_csv)

    order = ["raw", "re5", "re10", "re15"]
    missing = [k for k in order if k not in data]
    if missing:
        raise ValueError(f"Missing representations in CSV: {missing}. Found: {list(data.keys())}")

    precisions = [data[k][0] for k in order]
    recalls    = [data[k][1] for k in order]
    f1s        = [data[k][2] for k in order]

    save_barplot(order, precisions, "AE Classifier Precision", "Precision",
                 out_dir / "ae_precision_bar.png")
    save_barplot(order, recalls, "AE Classifier Recall", "Recall",
                 out_dir / "ae_recall_bar.png")
    save_barplot(order, f1s, "AE Classifier F1", "F1",
                 out_dir / "ae_f1_bar.png")

    return
#----------------------------------------------------------------


def mse_per_sample(model, X, batch_size=512):
    """
    X: np.ndarray shape (N, M)
    returns: errors shape (N,), one MSE value per datapoint
    """
    X = np.asarray(X, dtype=np.float32)
    X_rec = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - X_rec) ** 2, axis=1)    #average mse over all bytes per datapoint

def threshold_from_train_normal(model, X_train, y_train, percentile=95, batch_size=512):
    """
    Compute gamma ONLY from normal/control samples in training fold.
    y_train: 0=normal/control, 1=attack
    """
    X_norm = X_train[y_train == 0]  #select only control training samples

    #sanity check
    if X_norm.shape[0] == 0:
        raise ValueError("No normal samples in training fold -> cannot compute gamma.")

    errors = mse_per_sample(model, X_norm, batch_size=batch_size)
    gamma = float(np.percentile(errors, percentile)) #95% of the normal errors are ≤ gamma, sorts and sets threshold below top 5 highest
    return gamma

def ae_predict_labels(model, X, gamma, batch_size=512):
    """
    Classify X with given gamma.
    returns: y_pred (0/1), errors (per sample)
    """
    errors = mse_per_sample(model, X, batch_size=batch_size)
    y_pred = (errors > gamma).astype(int) #attack (1), if higher than threshold
    return y_pred, errors

def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1



def split_training_and_test(ds, labels, train_indices_fold, test_indices_fold):

    # separate features
    X_train = ds[train_indices_fold]
    X_test = ds[test_indices_fold]

    # separate labels
    y_train = labels[train_indices_fold]
    y_test = labels[test_indices_fold]

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, X_test, y_train, y_test



def execute_fold_ae(
    fold_idx,
    binary_numeric_labels,
    param,
    train_idx,
    test_idx,
    percentile=95,
    batch_size=512
):

    """
    Fully integrated fold evaluation:
    - load model for fold
    - compute gamma from TRAIN NORMAL ONLY
    - classify TEST
    - compute metrics
    """


    # ---- load features for this fold ----
    if param != 0:
        ds = np.load(f"datasets/re_bytes_{param}.npy")
        print(f"Experiment: Auto Encoder as classifier, fold {fold_idx}, scenario 1, RE{param}")
        model = keras.models.load_model(f"models/ae_fold{fold_idx}_re{param}.keras")
    else:
        ds = np.load(f"datasets/raw_bytes.npy")
        print(f"Experiment: Auto Encoder as Classifier, fold {fold_idx}, scenario 1, RAW")
        model = keras.models.load_model(f"models/ae_fold{fold_idx}_raw.keras")

    X_train, X_test, y_train, y_test = split_training_and_test(
        ds, binary_numeric_labels, train_idx, test_idx)

    gamma = threshold_from_train_normal(
        model, X_train, y_train, percentile=percentile, batch_size=batch_size
    )

    # classify TEST
    y_pred, _ = ae_predict_labels(model, X_test, gamma, batch_size=batch_size)

    # metrics
    prec, rec, f1 = precision_recall_f1(y_test, y_pred)
    return prec, rec, f1, gamma


def run_ae_for_scenario(
    prefix,
    global_label_encoder,
    keep_indices,
    param=0,
    percentile=95,
    batch_size=512
):

    labels = np.load(f"datasets/{prefix}_labels.npy")
    timestamps = np.load(f"datasets/{prefix}_timestamps.npy", allow_pickle=True)

    train_indices, test_indices = load_k_fold_results(f"k_fold_results/k_fold_s1_{prefix}.json")

    if param == 15:
        labels, timestamps = deduplicate_labels_and_timestamps(labels, timestamps, keep_indices)
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels = np.where(numeric_labels == 0, 0, 1)

    precisions, recalls, f1s, gammas = [], [], [], []

    k = len(train_indices)
    for fold_idx in range(k):
        if len(train_indices[fold_idx]) == 0 or len(test_indices[fold_idx]) == 0:
            continue

        try:
            p, r, f1, gamma = execute_fold_ae(
                fold_idx=fold_idx,
                binary_numeric_labels=binary_numeric_labels,
                param=param,
                train_idx=train_indices[fold_idx],
                test_idx=test_indices[fold_idx],
                percentile=percentile,
                batch_size=batch_size
            )
        except ValueError as e:

            print(f"[Fold {fold_idx}] skipped: {e}")
            continue

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        gammas.append(gamma)

        print(f"[Fold {fold_idx}] P={p:.4f} R={r:.4f} F1={f1:.4f} gamma={gamma:.6g}")

    if len(precisions) == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))




def run_ae_classifier_on_dataset(prefix, param, global_label_encoder):
    """
    Trains (if needed), runs k-fold AE classifier, and SAVES results to CSV instead of printing.
    Returns avg_p, avg_r, avg_f1.
    """

    train_and_save_models_classifier(prefix)


    if param != 0:
        # make sure the path includes .npy
        keep_indices = get_keep_indices_from_fold0_ae(f"datasets/re_bytes_{param}.npy")
        prefix_for_files = "re"   # because your labels/kfold files are re_labels.npy, k_fold_s1_re.json
    else:
        keep_indices = []
        prefix_for_files = "raw"

    avg_p, avg_r, avg_f1 = run_ae_for_scenario(
        prefix=prefix_for_files,
        global_label_encoder=global_label_encoder,
        keep_indices=keep_indices,
        param=param,
        percentile=95
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summary_path = out_dir / "ae_summary.csv"
    write_header = not summary_path.exists()

    representation = "raw" if param == 0 else f"re{param}"

    with summary_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["representation", "avg_precision", "avg_recall", "avg_f1"])
        w.writerow([representation, f"{avg_p:.6f}", f"{avg_r:.6f}", f"{avg_f1:.6f}"])

    return avg_p, avg_r, avg_f1


def run_experiment_ae_classifier(global_label_encoder):
    """
    Runs AE classifier on multiple prefixes.
    """
    prefixes = ["raw", "re5", "re10", "re15"]

    params=[0,5,10,15]
    results = {}
    for i in range(len(prefixes)):
        results[prefixes[i]] = run_ae_classifier_on_dataset(prefixes[i], params[i], global_label_encoder)
    return results

