import threading
import time
from pathlib import Path
import os




import tensorflow as tf
from tensorflow import keras

import numpy as np

from file_helper_t3 import load_k_fold_results


#returns the
def ds_training_and_test_from_fold(ds, labels, train_indices_fold, test_indices_fold):

    #extract datapoints for the training portion
    X_train = ds[train_indices_fold]

    X_test = ds[test_indices_fold]

    # remove attack data from test portion (can not be used for autoencoder validation)
    y_test = labels[test_indices_fold]
    test_mask = (y_test == "CONTROL")   #bit mask
    X_test = X_test[test_mask]

    # --- normalization: Bytes -> [0, 1] for testing, since the ae can work better with smaller range ---
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, X_test



# Train one AE per fold for a given representation (raw or re).
def train_ae_for_representation(
    base_model_path: str,   #path to untrained .keras model with correct input_dim
    kfold_json_path: str,   #path to k-fold indices json for scenario1
    labels_path: str,   #path to labels file
    bytes_path: str,    # path to preprocessed re_bytes or raw_bytes
    model_prefix: str,  #re or raw
):

    from measure_runtime import stop_ram_monitor, start_ram_monitor, bytes_to_mb

    # load base (untrained) model
    base_model = keras.models.load_model(base_model_path)

    # load data + folds
    training_indices, test_indices = load_k_fold_results(kfold_json_path)
    labels = np.load(labels_path)
    ds = np.load(bytes_path)  # shape (N, M)

    fold_runtimes = []
    fold_peak_ram = []

    for fold_idx in range(len(training_indices)):
        # start RAM tracker
        ram_handle = start_ram_monitor(interval=0.1)

        # measure runtime
        start_time = time.perf_counter()  # ⏱ start timing

        print(f"\n=== {model_prefix.upper()}: Fold {fold_idx} ===")

        train_idx = training_indices[fold_idx]
        test_idx = test_indices[fold_idx]

        # CONTROL-only + normalization
        X_train, X_val = ds_training_and_test_from_fold(ds, labels, train_idx, test_idx)
        print(f"\ntraining with {len(X_train)} datapoints")
        # fresh model for this fold
        model = keras.models.clone_model(base_model)
        model.set_weights(base_model.get_weights())
        model.compile(optimizer="adam", loss="mse")

        #stops the training if mse didn't improve for 5 epochs
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True,)]



        model.fit(
            X_train,     # inputs
            X_train,      # targets to reconstruct
            validation_data=(X_val, X_val), # val inputs, val targets
            epochs=100,
            callbacks=cb,
            verbose=1,
        )



        model.save(f"models/ae_fold{fold_idx}_{model_prefix}.keras")    #10 different models

        # measure runtime end
        end_time = time.perf_counter()
        runtime = end_time - start_time
        fold_runtimes.append(runtime)

        # measure ram end
        peak_rss = stop_ram_monitor(ram_handle)
        fold_peak_ram.append(peak_rss)

    avg_runtime = np.mean(fold_runtimes)
    avg_peak_ram=bytes_to_mb(np.max(fold_peak_ram))
    return avg_runtime, avg_peak_ram


def train_and_save_models_rt(prefix):
    base_dir = Path(__file__).resolve().parent

    if prefix == "raw":
        print("Training Models on RAW:\n")

        avg_runtime, avg_peak_ram = train_ae_for_representation(
            base_model_path=base_dir / "models" / "ae_untrained_466.keras",
            kfold_json_path=base_dir / "k_fold_results" / "k_fold_s1_raw.json",
            labels_path=base_dir / "datasets" / "raw_labels.npy",
            bytes_path=base_dir / "datasets" / "raw_bytes.npy",
            model_prefix="raw",
        )
    else:
        print("\nTraining Models on RE15:\n")

        avg_runtime, avg_peak_ram = train_ae_for_representation(
            base_model_path=base_dir / "models" / "ae_untrained_386.keras",
            kfold_json_path=base_dir / "k_fold_results" / "k_fold_s1_re.json",
            labels_path=base_dir / "datasets" / "re_labels.npy",
            bytes_path=base_dir / "datasets" / "re_bytes_15.npy",
            model_prefix="re",
        )

    return avg_runtime, avg_peak_ram


#gets k by checking length of training indices by fold!
def train_and_save_models():

    print("Training Models on RAW:\n")
    # RAW
    train_ae_for_representation(
        base_model_path="models/ae_untrained_466.keras",
        kfold_json_path="k_fold_results/k_fold_s1_raw.json",
        labels_path="datasets/raw_labels.npy",
        bytes_path="datasets/raw_bytes.npy",
        model_prefix="raw",
    )

    print("\nTraining Models on RE:\n")
    # RE
    train_ae_for_representation(
        base_model_path="models/ae_untrained_386.keras",
        kfold_json_path="k_fold_results/k_fold_s1_re.json",
        labels_path="datasets/re_labels.npy",
        bytes_path="datasets/re_bytes.npy",
        model_prefix="re",
    )

    return



############################feature extraction!

def extract_and_save_features_for_model(
        model_path: str, #eg models/ae_fold0_raw.keras
        bytes_path: str, #eg datasets/raw_bytes.npy
        out_path: str, #eg datasets/raw_features_fold0.npy
        batch_size: int = 128):
    """
    Load a trained AE, build encoder(latent), run it on the dataset (re_bytes/raw_bytes),
    and save the latent features to out_path (.npy).
    """
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)

    # Build encoder model: input -> latent layer
    latent_layer = model.get_layer("latent")

    # separates and extracts the Encoder portion of the ae
    encoder = keras.Model(inputs=model.input, outputs=latent_layer.output)

    # Load and normalize bytes
    print(f"Loading bytes from {bytes_path}")
    ds_all = np.load(bytes_path)          # shape (N, M), uint8
    ds_all = ds_all.astype("float32") / 255.0

    ds_all_batched = tf.data.Dataset.from_tensor_slices(ds_all).batch(batch_size)

    print(f"Extracting features for {ds_all.shape[0]} datapoints ...")
    features = encoder.predict(ds_all_batched, verbose=1)   # shape (N, bottleneck) - N = len(ds_all)

    print(f"Saving features to {out_path}")
    np.save(out_path, features)



def extract_features_for_all_folds(
    model_prefix: str, #re/raw
    bytes_path: str, # eg raw_bytes
    num_folds: int,
    out_dir: str = "datasets",
):
    from measure_runtime import start_ram_monitor, stop_ram_monitor, bytes_to_mb

    Path(out_dir).mkdir(exist_ok=True)

    fold_runtimes=[]
    fold_peak_ram=[]
    if model_prefix == "raw":
        model_prefix_ae="raw"
    else:
        model_prefix_ae="re"    #for rt measurements, re autoencoders will be trained on re15!
    for fold_idx in range(num_folds):
        model_path = f"models/ae_fold{fold_idx}_{model_prefix_ae}.keras"   #model trained for the specific fold (0 to k-1)
        out_path = f"{out_dir}/{model_prefix}_features_fold{fold_idx}.npy"

        ram_handle = start_ram_monitor(interval=0.1)

        # measure runtime
        start_time = time.perf_counter()  # ⏱ start timing

        extract_and_save_features_for_model(
            model_path=model_path,
            bytes_path=bytes_path,
            out_path=out_path,
            batch_size=128,
        )

        # measure runtime end
        end_time = time.perf_counter()
        runtime = end_time - start_time
        fold_runtimes.append(runtime)

        # measure ram end
        peak_rss = stop_ram_monitor(ram_handle)
        fold_peak_ram.append(peak_rss)
    avg_runtime = np.mean(fold_runtimes)
    avg_peak_ram = bytes_to_mb(np.max(fold_peak_ram))
    return avg_runtime, avg_peak_ram



#creae features for RAW and RE
def create_features_for_ds_rt(num_folds: int = 5, prefix="raw"):
    raw_path = Path("datasets/raw_bytes.npy")
    re_path = Path("datasets/re_bytes.npy")

    if not raw_path.exists() or not re_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found ({raw_path} or {re_path}). Make sure to extract them first."
        )

    if prefix=="raw":
        print("Using Autoencoder to create features for RAW bytes dataset\n")
        avg_runtime, avg_peak_ram=extract_features_for_all_folds(
        model_prefix="raw",
        bytes_path=str(raw_path),
        num_folds=num_folds,
        out_dir="datasets",
        )
        print("Feature extraction for RAW done!")
    else:
        print("\nUsing Autoencoder to create features for RE15 bytes dataset\n")
        avg_runtime, avg_peak_ram=extract_features_for_all_folds(
            model_prefix=f"re{15}",
            bytes_path=str(re_path),
            num_folds=num_folds,
            out_dir=f"datasets/re_bytes_{15}/",
        )
        print("Feature extraction for RE done!")

    return avg_runtime, avg_peak_ram


#creae features for RAW and RE
def create_features_for_ds(num_folds: int = 5):
    raw_path = Path("datasets/raw_bytes.npy")
    re_path = Path("datasets/re_bytes.npy")

    if not raw_path.exists() or not re_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found ({raw_path} or {re_path}). Make sure to extract them first."
        )

    print("Using Autoencoder to create features for RAW bytes dataset\n")
    avg_runtime, avg_peak_ram=extract_features_for_all_folds(
        model_prefix="raw",
        bytes_path=str(raw_path),
        num_folds=num_folds,
        out_dir="datasets",
    )
    print("Feature extraction for RAW done!")

    print("\nUsing Autoencoder to create features for RE bytes dataset\n")
    extract_features_for_all_folds(
        model_prefix="re",
        bytes_path=str(re_path),
        num_folds=num_folds,
        out_dir="datasets",
    )
    print("Feature extraction for RE done!")

    return avg_runtime, avg_peak_ram


#creae features for RAW and RE
def create_features_for_ds_task3def(num_folds: int = 5):

    p=[5,10,15]
    for i in p:
        re_path = Path(f"datasets/re_bytes_{i}.npy")
        os.makedirs(f"datasets/re_bytes_{i}/", exist_ok=True)

        if not re_path.exists():
            raise FileNotFoundError(
                f"Dataset files not found ({re_path}). Make sure to extract them first."
            )

        print(f"Using Autoencoder to create features for RE{i} bytes dataset\n")
        extract_features_for_all_folds(
            model_prefix=f"re{i}",
            bytes_path=str(re_path),
            num_folds=num_folds,
            out_dir=f"datasets/re_bytes_{i}/",
        )
        print(f"Feature extraction for RE{i} done!")