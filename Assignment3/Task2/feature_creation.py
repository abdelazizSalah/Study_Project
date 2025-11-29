from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np

#print: model parameters
def extract_and_print_features_to_file(ds_all_batched, is_raw):
    model_path = "models/model_dense_elu.keras"
    trainedModel = keras.models.load_model(model_path)



    encoder = keras.Model(trainedModel.input, trainedModel.get_layer("latent").output)
    features = encoder.predict(ds_all_batched, verbose=1)
    if is_raw:# works for tf.data.Dataset or np arrays
        filename = f"datasets/raw_features.npy"
    else:
        filename = f"datasets/re_features.npy"
    np.save(filename, features)




def created_features_for_ds():
    raw_path = Path("datasets/raw_bytes.npy")
    re_path = Path("datasets/re_bytes.npy")

    # --- Prüfung der RAW-Datei ---
    if not raw_path.exists():
        # Fehler ausgeben und Funktion beenden
        raise FileNotFoundError(
            f"Dataset files with raw bytes or re bytes not found {raw_path}. Make sure to extract them first."
        )
    #raw
    print("Using Autoencoder to create features for raw bytes dataset\n")
    raw_bytes = np.load(raw_path)
    ds_all = tf.data.Dataset.from_tensor_slices(raw_bytes)
    ds_all_batched = ds_all.batch(128)

    extract_and_print_features_to_file(ds_all_batched, 1)
    print("Feature extraction for raw done!")
    print("Using Autoencoder to create features for reverse engineering bytes dataset\n")
    re_bytes=np.load(re_path)
    ds_all = tf.data.Dataset.from_tensor_slices(re_bytes)
    ds_all_batched = ds_all.batch(128)

    extract_and_print_features_to_file(ds_all_batched, 0)
    print("Feature extraction for re done!")

    return
