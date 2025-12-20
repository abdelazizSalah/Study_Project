import numpy as np
from pathlib import Path
import os 
PATH = Path('../../DataSets/Assignment4_experiment_data/Assignment3DataGroup17/datasets')


def fast_deduplicate_and_save(input_path: str, output_path: str):
    """
    Fast row-wise deduplication using hashing + memory mapping.
    Keeps the first occurrence of each row and preserves order.
    """
    print(f"[+] Loading {input_path} (mmap)...")
    data = np.load(input_path, mmap_mode='r')

    print(f"[+] Computing row hashes...")
    # hash كل صف بدل np.unique(axis=0)
    row_hashes = np.fromiter(
        (hash(row.tobytes()) for row in data),
        dtype=np.int64,
        count=len(data)
    )

    print(f"[+] Removing duplicates...")
    _, unique_indices = np.unique(row_hashes, return_index=True)
    keep_indices = np.sort(unique_indices)

    dedup_data = np.asarray(data[keep_indices])  # materialize once

    print(
        f"[✓] {input_path}: "
        f"before={len(data)}, after={len(dedup_data)}, "
        f"removed={len(data) - len(dedup_data)}"
    )

    np.save(output_path, dedup_data)
    # save indicies as well
    np.save(output_path + '_indices.npy', keep_indices)
    print(f"[✓] Saved → {output_path}\n")
    print(f' length of dedup data: {len(dedup_data)}')
    print(f' length of keep indices: {len(keep_indices)}')


def main():
    for p in [5, 10, 15]:
        input_file = f"processed_packets_{p}.npy"
        output_file = f"processed_packets_{p}_dedup.npy"

        if not os.path.exists(input_file):
            print(f"[!] File not found: {input_file}")
            continue

        fast_deduplicate_and_save(input_file, output_file)


if __name__ == "__main__":
    main()
