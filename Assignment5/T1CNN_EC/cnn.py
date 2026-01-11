import csv
from pathlib import Path

import numpy as np
from h5py.h5fd import MEM_DRAW
from tensorflow import keras
from tensorflow.keras import layers
from file_helper_t3 import load_k_fold_results
from labels_helper import deduplicate_folds, encode_labels
from handling_re_bytes_integrated import get_keep_indices_from_fold0_ae


def build_cnn_classifier(
    M: int,
    dropout: float = 0.25,
    kernel_size: int = 7,
    filters=(8, 16, 32, 64),
    dense_units: int = 256,
    lr: float = 1e-3,
):
    """
    CNN with:
      - 4 blocks: (Conv -> BN -> ReLU -> Dropout) repeated twice per block
      - then 2 fully-connected layers (Dense + Dense softmax)
    Uses Adam + categorical cross-entropy.

    Input: (M, 1)
    Output: (2,) softmax => [P(normal), P(attack)]
    """
    assert len(filters) == 4, "Need exactly 4 conv blocks."

    inp = keras.Input(shape=(M, 1), name="bytes")

    x = inp
    for bi, f in enumerate(filters, start=1):
        # 2× (Conv -> BN -> Act -> Dropout)
        x = layers.Conv1D(f, kernel_size, padding="same", name=f"b{bi}_conv1")(x)
        x = layers.BatchNormalization(name=f"b{bi}_bn1")(x)
        x = layers.Activation("relu", name=f"b{bi}_relu1")(x)
        x = layers.Dropout(dropout, name=f"b{bi}_drop1")(x)

        x = layers.Conv1D(f, kernel_size, padding="same", name=f"b{bi}_conv2")(x)
        x = layers.BatchNormalization(name=f"b{bi}_bn2")(x)
        x = layers.Activation("relu", name=f"b{bi}_relu2")(x)
        x = layers.Dropout(dropout, name=f"b{bi}_drop2")(x)

        # Optional but very typical: downsample
        #x = layers.MaxPooling1D(pool_size=2, name=f"b{bi}_pool")(x)

    # Reduce time dimension -> vector
    x = layers.GlobalAveragePooling1D(name="gap")(x)

    # Fully-connected layer-every input neuron connected to all output layers
    x = layers.Dense(dense_units, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)   # not required by sheet, but fine
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(dropout, name="fc1_drop")(x)

    # Fully-connected layer 2 = output (Block 6)
    out = layers.Dense(2, activation="softmax", name="pred")(x)

    model = keras.Model(inp, out, name="cnn_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=[
            keras.metrics.CategoricalAccuracy(name="acc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
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
    # run on small portion of dataset, for demonstration:
    first_indices = np.arange(0, 1000)
    last_indices = np.arange(len(X_train) - 1000, len(X_train))
    selected_indices = np.concatenate((first_indices, last_indices))
    X_train = X_train[selected_indices]
    y_train = y_train[selected_indices]


    model = build_cnn_classifier(
        M=M,
        dropout=dropout,
        lr=lr,
        # you can expose these too if you want:
        # kernel_size=7,
        # filters=(32,64,128,256),
        # dense_units=256,
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


def run_cnn_for_scenario(
    scenario: int,
    prefix_for_files: str,   # "raw" or "re"
    global_label_encoder,
    keep_indices,
    param: int,
    M: int,
    epochs: int = 15,
    batch_size: int = 256,
    dropout: float = 0.25,
    lr: float = 1e-3,
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