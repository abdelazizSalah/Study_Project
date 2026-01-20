import numpy as np
from pathlib import Path

PATH = '../../DataSets/Assignment4_experiment_data/Assignment3DataGroup17/datasets'


def get_keep_indices_from_fold0(feature_dir: str, model_prefix: str) -> np.ndarray:
    """
    Load fold 0 features and compute indices to keep so that all duplicate
    rows are removed (only the first occurrence of each row is kept).
    """
    feature_dir = Path(feature_dir)
    fold0_path = feature_dir / f"{model_prefix}_features_fold0.npy"

    if not fold0_path.exists():
        raise FileNotFoundError(f"Feature file not found: {fold0_path}")

    feats0 = np.load(fold0_path)

    # unique over rows; index gives first position of each unique row
    _, unique_indices = np.unique(feats0, axis=0, return_index=True)

    # sort so we preserve original order
    keep_indices = np.sort(unique_indices)

    print(f"Fold 0: total samples = {len(feats0)}, "
          f"after dedup = {len(keep_indices)} "
          f"(removed {len(feats0) - len(keep_indices)} duplicates)")
    return keep_indices


# read re_bytes_5.npy, re_bytes_10.npy, re_bytes_15.npy, print unique, and count for each unique row
# then remove duplicates and save new files as re_bytes_5_dedup.npy, re_bytes_10_dedup.npy, re_bytes_15_dedup.npy
def remove_duplicates_from_preprocessed_files():
    p = [5, 10, 15]
    for i in p:
        input_file_path = f"{PATH}/re_bytes_{i}.npy"
        output_file_path = f"{PATH}/re_bytes_{i}_dedup.npy"
        data = np.load(input_file_path)
        print(f"Processing file: {input_file_path}")
        print(f"Total samples before dedup: {len(data)}")
        unique_data, indices = np.unique(data, axis=0, return_index=True)
        sorted_indices = np.sort(indices)
        dedup_data = data[sorted_indices]
        print(f"Total samples after dedup: {len(dedup_data)}")
        np.save(output_file_path, dedup_data)
        print(f"Deduplicated data saved to: {output_file_path}\n")
    return


if __name__ == "__main__":
    # show unique data with count for each row before processing and after processing
    # for i in [5, 10, 15]:
    input_file_path = f"{PATH}/re_labels.npy"
    # read labels
    print(f"Loading labels from: {input_file_path}")
    labels = np.load(input_file_path)
    print('labeled successfully loaded')
    # print labels count before dedup
    unique, counts = np.unique(labels, return_counts=True)
    print(f"unique labels before dedup: {dict(zip(unique, counts))}\n")
    print(len(labels))

    for p in [5,10,15]:
        input_file_path = f'processed_packets_{p}.npy'
        # load processed packets
        print(f"Loading processed packets from: {input_file_path}")
        packets = np.load(input_file_path)
        print('processed packets successfully loaded')
        # print its length
        print(f"Total processed packets before dedup: {len(packets)}")
