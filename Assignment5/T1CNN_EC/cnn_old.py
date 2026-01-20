import csv
import json
from itertools import product
from pathlib import Path
import numpy as np
from h5py.h5fd import MEM_DRAW
from tensorflow import keras
from tensorflow.keras import layers
from file_helper_t3 import load_k_fold_results
from labels_helper import deduplicate_folds, encode_labels
from handling_re_bytes_integrated import get_keep_indices_from_fold0_ae
from tensorflow.keras import regularizers
import pandas as pd
import matplotlib.pyplot as plt



def build_cnn_classifier(
    M: int,
    dropout: float = 0.25,
    kernel_size: int = 7,
    filters=(8, 16, 32, 64),
    dense_units: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
):
    reg = regularizers.l2(weight_decay) if weight_decay and weight_decay > 0 else None

    inp = keras.Input(shape=(M, 1), name="bytes")
    x = inp

    for bi, f in enumerate(filters, start=1):
        x = layers.Conv1D(
            f, kernel_size, padding="same",
            kernel_regularizer=reg,  # <-- add
            name=f"b{bi}_conv1"
        )(x)
        x = layers.BatchNormalization(name=f"b{bi}_bn1")(x)
        x = layers.Activation("relu", name=f"b{bi}_relu1")(x)
        x = layers.Dropout(dropout, name=f"b{bi}_drop1")(x)

        x = layers.Conv1D(
            f, kernel_size, padding="same",
            kernel_regularizer=reg,  # <-- add
            name=f"b{bi}_conv2"
        )(x)
        x = layers.BatchNormalization(name=f"b{bi}_bn2")(x)
        x = layers.Activation("relu", name=f"b{bi}_relu2")(x)
        x = layers.Dropout(dropout, name=f"b{bi}_drop2")(x)

    # data reduction
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    x = layers.Dense(dense_units, kernel_regularizer=reg, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout, name="fc1_drop")(x)

    out = layers.Dense(2, activation="softmax", kernel_regularizer=reg, name="pred")(x)  # optional

    model = keras.Model(inp, out, name="cnn_classifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    return model



def train_cnn(
    model: keras.Model,
    X_train: np.ndarray,
    y_train01: np.ndarray,
    epochs: int = 15,
    batch_size: int = 256,
    validation_split: float = 0.1,
    verbose: int = 0,
):
    """
    Train an already-built + compiled CNN model.

    X_train: (N, M, 1) float32
    y_train01: (N,) with values 0/1
    """
    y_train_oh = keras.utils.to_categorical(y_train01, num_classes=2)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train,
        y_train_oh,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=True,
        verbose=verbose,
        callbacks=callbacks,
    )
    return history

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


def evaluate_cnn(
    model: keras.Model,
    X_test: np.ndarray,
    y_test01: np.ndarray,
    batch_size: int = 256,
):
    """
    Evaluate a trained CNN model.

    X_test: (N, M, 1) float32
    y_test01: (N,) with values 0/1

    Returns: (precision, recall, f1, y_pred)
    """
    probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    y_pred = probs.argmax(axis=1)

    # uses your existing function
    precision, recall, f1 = precision_recall_f1(y_test01, y_pred)
    return float(precision), float(recall), float(f1), y_pred


def split_training_and_test_cnn(ds, labels, train_indices_fold, test_indices_fold):
    #normalize bytes
    X_train = ds[train_indices_fold].astype("float32") / 255.0
    X_test  = ds[test_indices_fold].astype("float32") / 255.0

    # add channel dimension for Conv1D: (N, M) -> (N, M, 1)
    X_train = X_train[..., None]
    X_test  = X_test[..., None]

    y_train = labels[train_indices_fold]
    y_test  = labels[test_indices_fold]
    return X_train, X_test, y_train, y_test


#used for the RE splits
def oversample_minority(X, y01, target_pos_ratio=0.3, seed=0):
    rng = np.random.default_rng(seed)
    pos = np.where(y01 == 1)[0]
    neg = np.where(y01 == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y01

    # desired number of positives
    n_total = len(y01)
    n_pos_target = int(target_pos_ratio * n_total)
    if n_pos_target <= len(pos):
        return X, y01

    extra = rng.choice(pos, size=(n_pos_target - len(pos)), replace=True)
    idx = np.concatenate([np.arange(n_total), extra])
    rng.shuffle(idx)
    return X[idx], y01[idx]



def execute_fold_cnn(
    fold_idx: int,
    binary_numeric_labels: np.ndarray,
    scenario: int,
    param: int,
    train_idx,
    test_idx,
    M: int,
    epochs: int = 15,
    batch_size: int = 256,
    dropout: float = 0.25,
    lr: float = 1e-3,
    kernel_size: int = 7,
    filters=(8, 16, 32, 64),
    dense_units: int = 256,
    weight_decay: float = 0.0,
):

    # ---- load dataset ----
    if param != 0:
        ds = np.load(f"datasets/re_bytes_{param}.npy")
        print(f"Experiment: CNN classifier, fold {fold_idx}, scenario {scenario}, RE{param}")
    else:
        ds = np.load("datasets/raw_bytes.npy")
        print(f"Experiment: CNN classifier, fold {fold_idx}, scenario {scenario}, RAW")

    # split -> scale -> add channel dim (N,M,1)
    X_train, X_test, y_train, y_test = split_training_and_test_cnn(
        ds, binary_numeric_labels, np.array(train_idx), np.array(test_idx)
    )

    # optional: skip fold if no attack in train
    if np.sum(y_train == 1) == 0:
        print(f"[Fold {fold_idx}] skipped: no attack samples in TRAIN")
        return 0.0, 0.0, 0.0

    X_train, y_train = oversample_minority(X_train, y_train, target_pos_ratio=0.3, seed=fold_idx)

    # run on small portion of dataset, for demonstration:
    #first_indices = np.arange(0, 10000)
    #last_indices = np.arange(len(X_train) - 10000, len(X_train))
    #selected_indices = np.concatenate((first_indices, last_indices))
    #X_train = X_train[selected_indices]
    #y_train = y_train[selected_indices]

    model = build_cnn_classifier(
        M=M,
        dropout=dropout,
        lr=lr,
        kernel_size=kernel_size,
        filters=filters,
        dense_units=dense_units,
        weight_decay=weight_decay
    )


    train_cnn(
        model=model,
        X_train=X_train,
        y_train01=y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # evaluate on test
    prec, rec, f1, _ = evaluate_cnn(
        model=model,
        X_test=X_test,
        y_test01=y_test,
        batch_size=batch_size,
    )

    return prec, rec, f1



def grid_search_cnn_scenario(
    scenario: int,
    prefix_for_files: str,  #raw or reX
    global_label_encoder,
    keep_indices,
    param: int,
    M: int,
    # fixed training settings
    epochs: int = 10,
    # grid
    grid_dropout=(0.2, 0.3),    #dropout rate ()higher means more dropouts)
    grid_lr=(1e-3, 5e-4),   #learning rate for adam optimizer
    grid_kernel_size=(5, 7),    #kernel size fof each Conv1D
    grid_filters=((8,16,32,64), (16,32,64,128)),
    grid_dense_units=(128, 256),    #numbe of neurons in the fully connected layer
    grid_batch_size = (128, 256),
    grid_weight_decay = (0.0, 1e-4, 1e-3)
):
    """
    Runs grid-search (configs × folds) and stores:
      1) a CSV with all tried configs + avg metrics
      2) a JSON with the best config + best scores

    Filenames include prefix_for_files, param and scenario.

    Returns: best_cfg, best_scores, all_results (list of dicts)
    """

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # filenames include prefix, param, scenario
    prefix_tag = prefix_for_files  # "raw" or "re"
    param_tag = f"p{param}"
    scen_tag = f"s{scenario}"
    base = f"cnn_grid_{prefix_tag}_{param_tag}_{scen_tag}"

    all_csv_path = out_dir / f"{base}_all.csv"
    best_json_path = out_dir / f"{base}_best.json"

    # load labels + folds once
    labels = np.load(f"datasets/{prefix_for_files}_labels.npy")
    train_indices, test_indices = load_k_fold_results(
        f"k_fold_results/k_fold_s{scenario}_{prefix_for_files}.json"
    )

    # your UPDATED deduplicate_folds: filters indices in original index space
    if param != 0:
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)

    numeric_labels = encode_labels(global_label_encoder, labels)
    y01 = np.where(numeric_labels == 0, 0, 1)

    all_results = []
    best = None

    # header for the "all results" CSV
    with all_csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "prefix_for_files", "param", "M", "epochs",
            "dropout", "lr", "kernel_size", "filters", "dense_units",
            "batch_size", "weight_decay",
            "avg_precision", "avg_recall", "avg_f1",
        ])

    for dropout, lr, ks, filt, du, bs, wd in product(
        grid_dropout, grid_lr, grid_kernel_size, grid_filters,
        grid_dense_units, grid_batch_size, grid_weight_decay
    ):
        precisions, recalls, f1s = [], [], []

        for fold_idx in range(len(train_indices)):
            if len(train_indices[fold_idx]) == 0 or len(test_indices[fold_idx]) == 0:
                continue

            p, r, f1 = execute_fold_cnn(
                fold_idx=fold_idx,
                binary_numeric_labels=y01,
                scenario=scenario,
                param=param,
                train_idx=train_indices[fold_idx],
                test_idx=test_indices[fold_idx],
                M=M,
                epochs=epochs,
                batch_size=bs,
                dropout=dropout,
                lr=lr,
                kernel_size=ks,
                filters=filt,
                dense_units=du,
                weight_decay=wd,
            )
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        avg_p = float(np.mean(precisions)) if precisions else 0.0
        avg_r = float(np.mean(recalls)) if recalls else 0.0
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0

        cfg = {
            "dropout": float(dropout),
            "lr": float(lr),
            "kernel_size": int(ks),
            "filters": tuple(int(x) for x in filt),
            "dense_units": int(du),
            "batch_size": int(bs),
            "weight_decay": float(wd),
        }
        row = {
            "cfg": cfg,
            "avg_precision": avg_p,
            "avg_recall": avg_r,
            "avg_f1": avg_f1,
        }
        all_results.append(row)

        # append this config to CSV
        with all_csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                scenario, prefix_for_files, param, M, epochs,
                cfg["dropout"], cfg["lr"], cfg["kernel_size"], str(cfg["filters"]),
                cfg["dense_units"], cfg["batch_size"], cfg["weight_decay"],
                f"{avg_p:.6f}", f"{avg_r:.6f}", f"{avg_f1:.6f}",
            ])

        print(f"[grid] cfg={cfg} -> F1={avg_f1:.4f}")

        if best is None or avg_f1 > best["avg_f1"]:
            best = row
            payload = {
                "scenario": scenario,
                "prefix_for_files": prefix_for_files,
                "param": param,
                "M": M,
                "epochs": epochs,
                "best_cfg": best["cfg"],
                "best_scores": {
                    "avg_precision": best["avg_precision"],
                    "avg_recall": best["avg_recall"],
                    "avg_f1": best["avg_f1"],
                },
            }
            best_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        keras.backend.clear_session()

    # final write (in case it never updated inside loop due to no folds)
    if best is None:
        best = {"cfg": None, "avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0}
        payload = {
            "scenario": scenario,
            "prefix_for_files": prefix_for_files,
            "param": param,
            "M": M,
            "epochs": epochs,
            "best_cfg": None,
            "best_scores": {"avg_precision": 0.0, "avg_recall": 0.0, "avg_f1": 0.0},
        }
        best_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return best["cfg"], (best["avg_precision"], best["avg_recall"], best["avg_f1"]), all_results








#without grid search
def run_cnn_for_scenario(
    scenario: int,
    prefix_for_files: str,   # "raw" or "re"
    global_label_encoder,
    keep_indices,
    param: int,
    M: int,
    epochs: int = 15,
    batch_size: int = 256,

    # model/training hyperparams
    dropout: float = 0.25,
    lr: float = 1e-3,
    kernel_size: int = 7,
    filters=(8, 16, 32, 64),
    dense_units: int = 256,
    weight_decay: float = 0.0

):
    """
    Runs k-fold CNN evaluation for one scenario (2 or 3) and one representation (raw/re{p}).
    Returns avg_precision, avg_recall, avg_f1.
    """

    labels = np.load(f"datasets/{prefix_for_files}_labels.npy")

    train_indices, test_indices = load_k_fold_results(
        f"k_fold_results/k_fold_s{scenario}_{prefix_for_files}.json"
    )

    if param != 0:
        train_indices, test_indices = deduplicate_folds(train_indices, test_indices, keep_indices)

    numeric_labels = encode_labels(global_label_encoder, labels)
    binary_numeric_labels = np.where(numeric_labels == 0, 0, 1)

    precisions, recalls, f1s = [], [], []

    k = len(train_indices)
    for fold_idx in range(k):
        if len(train_indices[fold_idx]) == 0 or len(test_indices[fold_idx]) == 0:
            continue

        p, r, f1 = execute_fold_cnn(
            fold_idx=fold_idx,
            binary_numeric_labels=binary_numeric_labels,
            scenario=scenario,
            param=param,
            train_idx=train_indices[fold_idx],
            test_idx=test_indices[fold_idx],
            M=M,
            epochs=epochs,
            batch_size=batch_size,
            dropout=dropout,
            lr=lr,
            kernel_size=kernel_size,
            filters=filters,
            dense_units=dense_units,
            weight_decay=weight_decay
        )

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        print(f"[Fold {fold_idx}] P={p:.4f} R={r:.4f} F1={f1:.4f}")

    if len(precisions) == 0:
        return 0.0, 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))



def run_cnn_classifier_on_dataset(
    scenario: int,
    prefix: str,               # "raw", "re5", "re10", "re15" (only used for naming)
    param: int,                # 0, 5, 10, 15
    global_label_encoder,
    M: int,
    out_csv: str = "results/cnn_summary.csv",
    do_grid=True
):
    """
    Runs CNN for ONE scenario and ONE representation, appends to CSV.
    """

    if param != 0:
        keep_indices = get_keep_indices_from_fold0_ae(f"datasets/re_bytes_{param}.npy")
        prefix_for_files = "re"   # because your labels/kfold files are re_labels.npy, k_fold_s{scenario}_re.json
    else:
        keep_indices = []
        prefix_for_files = "raw"

    if do_grid:
        best_cfg, best_scores, all_rows = grid_search_cnn_scenario(
            scenario=scenario,
            prefix_for_files=prefix_for_files,
            global_label_encoder=global_label_encoder,
            keep_indices=keep_indices,
            param=param,
            M=M,
            epochs=10,  # smaller for search
        )

        # run once more with best config
        avg_p, avg_r, avg_f1 = run_cnn_for_scenario(
            scenario=scenario,
            prefix_for_files=prefix_for_files,
            global_label_encoder=global_label_encoder,
            keep_indices=keep_indices,
            param=param,
            M=M,
            epochs=15,
            batch_size=best_cfg["batch_size"],
            dropout=best_cfg["dropout"],
            lr=best_cfg["lr"],
            kernel_size=best_cfg["kernel_size"],
            filters=best_cfg["filters"],
            dense_units=best_cfg["dense_units"],
            weight_decay=best_cfg["weight_decay"],
        )
    else:
        #no grid search
        avg_p, avg_r, avg_f1 = run_cnn_for_scenario(
            scenario=scenario,
            prefix_for_files=prefix_for_files,
            global_label_encoder=global_label_encoder,
            keep_indices=keep_indices,
            param=param,
            M=M,
        )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    summary_path = Path(out_csv)
    write_header = not summary_path.exists()

    representation = "raw" if param == 0 else f"re{param}"

    with summary_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["scenario", "representation", "M", "avg_precision", "avg_recall", "avg_f1"])
        w.writerow([scenario, representation, M, f"{avg_p:.6f}", f"{avg_r:.6f}", f"{avg_f1:.6f}"])

    return avg_p, avg_r, avg_f1


def run_experiment_cnn_classifier(global_label_encoder, M_raw,M_re):
    """
    Runs CNN classifier:
      - scenarios 2 and 3
      - representations raw, re5, re10, re15
    Appends all results to results/cnn_summary.csv
    """
    prefixes = ["raw", "re5", "re10", "re15"]
    params = [0, 5, 10, 15]

    results = {}
    for scenario in (2, 3):
        results[scenario] = {}
        for prefix, param in zip(prefixes, params):
            if param==0:
                M=M_raw
            else:
                M=M_re
            results[scenario][prefix] = run_cnn_classifier_on_dataset(
                scenario=scenario,
                prefix=prefix,
                param=param,
                global_label_encoder=global_label_encoder,
                M=M,
            )

    return results



#####plots:

def plot_cnn_summary_grouped(
    csv_path="results/cnn_summary.csv",
    out_dir="results",
):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # expected columns: scenario, representation, avg_precision, avg_recall, avg_f1
    required = {"scenario", "representation", "avg_precision", "avg_recall", "avg_f1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}. Found: {list(df.columns)}")

    # Keep only scenarios 2 and 3
    df["scenario"] = pd.to_numeric(df["scenario"], errors="coerce").astype("Int64")
    df = df[df["scenario"].isin([2, 3])].copy()
    if df.empty:
        raise ValueError("No rows for scenario 2/3 found in the CSV.")

    # Force order and make sure all reps exist
    rep_order = ["raw", "re5", "re10", "re15"]
    df["representation"] = df["representation"].astype(str).str.strip()
    df["representation"] = pd.Categorical(df["representation"], categories=rep_order, ordered=True)

    # Aggregate just in case you have multiple rows per (scenario, representation)
    agg = (
        df.groupby(["scenario", "representation"], as_index=False)[["avg_precision", "avg_recall", "avg_f1"]]
          .mean()
    )

    # Pivot to have columns = scenarios, rows = representations
    def make_plot(metric_col: str, title: str, ylabel: str, filename: str):
        piv = agg.pivot(index="representation", columns="scenario", values=metric_col).reindex(rep_order)

        # Check presence of both scenarios
        for s in [2, 3]:
            if s not in piv.columns:
                raise ValueError(f"Scenario {s} missing for metric {metric_col}.")

        x = np.arange(len(rep_order))
        width = 0.38  # bar width

        fig, ax = plt.subplots()

        # matplotlib will choose distinct default colors automatically
        ax.bar(x - width/2, piv[2].values, width, label="Scenario 2")
        ax.bar(x + width/2, piv[3].values, width, label="Scenario 3")

        ax.set_xticks(x)
        ax.set_xticklabels(rep_order)
        ax.set_ylim(0.0, 1.08)
        ax.set_title(title, pad=28)
        ax.set_xlabel("Representation")
        ax.set_ylabel(ylabel)
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.02),  # x=right edge, y=just above axes
            ncol=2,  # show Scenario 2 and 3 side-by-side
            frameon=True,
            borderaxespad=0.0
        )

        # value labels
        for xi, v2, v3 in zip(x, piv[2].values, piv[3].values):
            if pd.notna(v2):
                ax.text(xi - width/2, float(v2) + 0.01, f"{float(v2):.3f}", ha="center", va="bottom")
            if pd.notna(v3):
                ax.text(xi + width/2, float(v3) + 0.01, f"{float(v3):.3f}", ha="center", va="bottom")

        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    make_plot("avg_precision", "CNN Precision", "Precision", "cnn_precision_bar.png")
    make_plot("avg_recall",    "CNN Recall",    "Recall",    "cnn_recall_bar.png")
    make_plot("avg_f1",        "CNN F1",        "F1",        "cnn_f1_bar.png")

    print(f"Saved plots to: {out_dir.resolve()}")
