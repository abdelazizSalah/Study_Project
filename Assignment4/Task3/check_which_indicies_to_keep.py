import numpy as np
from pathlib import Path

def get_keep_indices_from_fold0(feature_dir: str, model_prefix: str) -> np.ndarray:
    """
    Load fold 0 features and compute indices to keep so that all duplicate
    rows are removed (only the first occurrence of each row is kept).
    """
    feature_dir = Path(feature_dir)
    fold0_path = feature_dir / f"{model_prefix}.npy"

    if not fold0_path.exists():
        raise FileNotFoundError(f"Feature file not found: {fold0_path}")
    print(f"Loading features from: {fold0_path}")
    feats0 = np.load(fold0_path)

    # unique over rows; index gives first position of each unique row
    print(f"Computing unique rows for {model_prefix}...")
    _, unique_indices = np.unique(feats0, axis=0, return_index=True)

    # sort so we preserve original order
    print(f"Computing keep indices for {model_prefix}...")
    keep_indices = np.sort(unique_indices)

    print(f"Fold 0: total samples = {len(feats0)}, "
          f"after dedup = {len(keep_indices)} "
          f"(removed {len(feats0) - len(keep_indices)} duplicates)")
    return keep_indices


# read processed_packets_5.npy, processed_packets_10.npy, processed_packets_15.npy, then use the above function to get keep indicies, then save indicies to files
for p in [5, 10, 15]:
    print(f"Processing processed_packets_{p}.npy")
    model_prefix = f'processed_packets_{p}'
    keep_indices = get_keep_indices_from_fold0('./', model_prefix)
    indices_output_path = f'processed_packets_{p}_dedup_indices.npy'
    np.save(indices_output_path, keep_indices)
    print(f"Saved keep indices to: {indices_output_path}\n")
    
