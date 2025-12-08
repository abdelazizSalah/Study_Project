from pathlib import Path
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from file_helper_t3 import load_k_fold_results


def build_ics_ae_dense(
    input_dim: int,
    bottleneck: int = 32,
    hidden_dims=(256, 128),
    activation: str = "relu",
    noise_std: float = 0.02,
    bottleneck_l1: float = 0.0,
) -> keras.Model:
    """
    Autoencoder für ICS-Pakete (bytes als Vektor, vorher auf [0,1] skalieren).

    - Encoder: GaussianNoise -> Dense(hidden_dims...) -> Bottleneck
    - Decoder: symmetrisch zurück auf input_dim
    - Output: sigmoid (passt zu [0,1] Skala), Loss: MSE
    """

    inp = keras.Input(shape=(input_dim,), name="bytes")

    # Rauschen für Denoising-Effekt (nur im Training aktiv)
    x = layers.GaussianNoise(noise_std, name="noise")(inp)

    # ENCODER
    for i, h in enumerate(hidden_dims):
        x = layers.Dense(h, activation=None, name=f"enc{i+1}_dense")(x)
        x = layers.BatchNormalization(name=f"enc{i+1}_bn")(x)
        x = layers.Activation(activation, name=f"enc{i+1}_act")(x)
        x = layers.Dropout(0.1, name=f"enc{i+1}_drop")(x)

    # Bottleneck / Latent-Space
    z = layers.Dense(
        bottleneck,
        activation=None,
        name="latent",
        kernel_regularizer=regularizers.l1(bottleneck_l1) if bottleneck_l1 > 0 else None,
    )(x)

    # DECODER (symmetrisch)
    x = z
    for i, h in enumerate(reversed(hidden_dims)):
        x = layers.Dense(h, activation=None, name=f"dec{i+1}_dense")(x)
        x = layers.BatchNormalization(name=f"dec{i+1}_bn")(x)
        x = layers.Activation(activation, name=f"dec{i+1}_act")(x)

    out = layers.Dense(input_dim, activation="sigmoid", name="recon")(x)

    model = keras.Model(inp, out, name="ICS_AE_Dense")
    return model




def create_model():
    # learning rate = size of single weight update step
    # higher lr -> faster training, lr too high -> training unstable



    M = 386  # dein max. Packet-Size
    model = build_ics_ae_dense(
        input_dim=M,
        bottleneck=32,  # ggf. 16 / 64 ausprobieren
        hidden_dims=(256, 128),  # minimal größer als vorher
        activation="relu",
        noise_std=0.02,
        bottleneck_l1=1e-5  # oder 0.0, wenn du kein L1 willst
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )
    model.save("models/ae_untrained_386.keras")
    return



############################creation and training of models (one for each fold)!


def ds_training_and_test_from_fold(ds, labels, train_indices_fold, test_indices_fold):

    # FEATURES AUFTEILEN
    X_train = ds[train_indices_fold]


    X_test = ds[test_indices_fold]

    # attack data aus test entfernen (Kontraproduktiv für autoencoder training)
    y_test = labels[test_indices_fold]
    test_mask = (y_test == "CONTROL")
    X_test = X_test[test_mask]

    # --- Normalisierung: Bytes -> [0, 1] ---
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    return X_train, X_test


def train_ae_for_representation(
    base_model_path: str,
    kfold_json_path: str,
    labels_path: str,
    bytes_path: str,
    model_prefix: str,
):
    """
    Train one AE per fold for a given representation (raw or re).

    base_model_path:   path to untrained .keras model (correct input_dim)
    kfold_json_path:   path to k-fold indices json
    labels_path:       npy file with labels (shape: (N,))
    bytes_path:        npy file with byte matrix (shape: (N, M))
    model_prefix:      prefix for saving trained models, e.g. 'raw' or 're'
    """
    # load base (untrained) model
    base_model = keras.models.load_model(base_model_path)

    # load data + folds
    training_indices, test_indices = load_k_fold_results(kfold_json_path)
    labels = np.load(labels_path)
    ds = np.load(bytes_path)  # shape (N, M)

    for fold_idx in range(len(training_indices)):
        print(f"\n=== {model_prefix.upper()}: Fold {fold_idx} ===")

        train_idx = training_indices[fold_idx]
        test_idx = test_indices[fold_idx]

        # CONTROL-only + normalization
        X_train, X_val = ds_training_and_test_from_fold(ds, labels, train_idx, test_idx)
        print(f"\ntraining with {len(X_train)} datapoints")
        # fresh model for this fold
        model = keras.models.clone_model(base_model)
        model.set_weights(base_model.get_weights())
        model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

        cb = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
            )
        ]

        model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val),
            epochs=100,
            callbacks=cb,
            verbose=1,
        )

        model.save(f"models/ae_fold{fold_idx}_{model_prefix}.keras")    #10 different models


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

def extract_and_save_features_for_model(model_path: str, bytes_path: str, out_path: str, batch_size: int = 128):
    """
    Load a trained AE, build encoder(latent), run it on the *whole* dataset,
    and save the latent features to out_path (.npy).
    """
    print(f"\nLoading model: {model_path}")
    model = keras.models.load_model(model_path)

    # Build encoder model: input -> latent layer
    latent_layer = model.get_layer("latent")
    encoder = keras.Model(inputs=model.input, outputs=latent_layer.output)

    # Load and normalize bytes
    print(f"Loading bytes from {bytes_path}")
    ds_all = np.load(bytes_path)          # shape (N, M), uint8
    ds_all = ds_all.astype("float32") / 255.0

    ds_all_batched = tf.data.Dataset.from_tensor_slices(ds_all).batch(batch_size)

    print(f"Extracting features for {ds_all.shape[0]} datapoints ...")
    features = encoder.predict(ds_all_batched, verbose=1)   # shape (N, bottleneck)

    print(f"Saving features to {out_path}")
    np.save(out_path, features)



def extract_features_for_all_folds(
    model_prefix: str,
    bytes_path: str,
    num_folds: int,
    out_dir: str = "datasets",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if model_prefix == "raw":
        model_prefix_ae="raw"
    else:
        model_prefix_ae="re"    #for re5, re10, re15 and re (size will be the same, same autoencoders per fold)
    for fold_idx in range(num_folds):
        model_path = f"models/ae_fold{fold_idx}_{model_prefix_ae}.keras"   #model trained for the specific fold (0 to k-1)
        out_path = f"{out_dir}/{model_prefix}_features_fold{fold_idx}.npy"

        extract_and_save_features_for_model(
            model_path=model_path,
            bytes_path=bytes_path,
            out_path=out_path,
            batch_size=128,
        )
    return



#creae features for RAW and RE
def create_features_for_ds(num_folds: int = 5):
    raw_path = Path("datasets/raw_bytes.npy")
    re_path = Path("datasets/re_bytes.npy")

    if not raw_path.exists() or not re_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found ({raw_path} or {re_path}). Make sure to extract them first."
        )

    print("Using Autoencoder to create features for RAW bytes dataset\n")
    extract_features_for_all_folds(
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