
from scapy.all import rdpcap
import numpy as np
from pathlib import Path
import tensorflow as tf
from scapy.error import Scapy_Exception

from Anna_Code.file_helper import list_files_by_filetype


def store_dataset_in_file(path, packet_length_M, is_attack_set):

    files = list_files_by_filetype(path, "pcap")
    matrix = create_matrix_from_pcaps(files, packet_length_M)

    #create training data (np matrix)
    ds_all, ds_train, ds_test, X_train, X_test = make_datasets(matrix)
    # save ds
    data = np.concatenate(list(ds_all.as_numpy_iterator()))
    if(is_attack_set):
        np.save("dataset_attack.npy", data)
    else:
        np.save("dataset_normal.npy", data)
    return


#input: single packet in binary, length M
#output: array for packet with required length
def to_fixed_bytes(raw, M) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)    #convert bytes to numpy array
    # truncate if longer than M
    if arr.size >= M:
        return arr[:M]

    #pad if smaller than M
    out = np.zeros(M, dtype=np.uint8) #array of length M filled with zeros
    out[:arr.size] = arr   #replace first part with original array
    return out


#input: pcap file
#output: packet list for that file
def read_pcap_bytes(pcap_file, M):
    pkts = rdpcap(str(pcap_file))
    packet_arrays_file=[]
    for p in pkts:
        raw=bytes(p)
        packet_arrays_file.append(to_fixed_bytes(raw, M))
    return packet_arrays_file


#input: list of pcap files, byte length M
#output: matrix for autoencoder
def create_matrix_from_pcaps(files, M):
    rows = []
    for file in files:
        try:
            packet_list = read_pcap_bytes(file, M)  # list of np arrays
        except Exception as e:
            print(f"[!] Unexpected error reading {file} - it might be empty: {e}")
            continue

        if not packet_list:
            print(f"[!] No packets found in {file}, skipping.")
            continue

        rows.extend(packet_list)

    if not rows:
        raise ValueError("No valid packets found in any of the provided PCAP files.")

    #stack list of arrays into single matrix
    #before: [  8,   0,  39,  46, 207, 244,   0,  27,  27,  23, 248, 130,   ...]
    packet_matrix = np.stack(rows).astype(np.float32) / 255.0  # normalize to [0,1], model expects values between 0 and 1

    #after: [8/255, 0.000, 39/255, ...]
    return packet_matrix


#assume unsupervised (no labels)
#train ratio 0.9 => 10% validation, 90% training
#batch size effects speed and resource consumption!
def make_datasets(matrix, batch_size = 128, train_ratio: float = 0.9):
    # Compute split index
    n_train = int(train_ratio * matrix.shape[0])

    # Split into train and validation
    X_train, X_test = matrix[:n_train], matrix[n_train:]

    # Create TensorFlow Datasets (unsupervised)
    ds_train = (
        tf.data.Dataset
        .from_tensor_slices((X_train, X_train))   # input = output for autoencoder
        .shuffle(len(X_train))                    # randomize sample order
        .batch(batch_size)                        # group into batches
    )

    ds_test = (
        tf.data.Dataset
        .from_tensor_slices((X_test, X_test))
        .batch(batch_size)
    )

    # Combine train and test into one dataset (e.g., for feature extraction)
    ds_all = (
        tf.data.Dataset
        .from_tensor_slices(matrix)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds_all, ds_train, ds_test, X_train, X_test
