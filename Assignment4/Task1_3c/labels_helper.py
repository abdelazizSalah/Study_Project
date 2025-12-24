
from collections import defaultdict
import numpy as np

"""
operations on the dataset based on the label array and ds array
"""

def get_amount_of_different_attack_types(labels):
    unique, counts = np.unique(labels, return_counts=True)

    return len(unique)


def split_attack_control(labels):
    control_indices = np.where(labels == "CONTROL")[0].tolist()
    attack_indices = np.where(labels != "CONTROL")[0].tolist()

    return control_indices,attack_indices


#returns list of all indices that correspond to a certain attack type
def get_indices_for_attack_type(labels, attack_type):
    return [i for i, lab in enumerate(labels) if lab == attack_type]


def get_attack_idx_by_type(labels):
    attack_idx_by_type = defaultdict(list)  #defaultdict: automatically creates a new list, when a new key is used

    for i, lab in enumerate(labels):
        if lab != "CONTROL":
            attack_idx_by_type[lab].append(i)

    return attack_idx_by_type               #{
                                            #   "ReactorOn": [idx1, idx2, idx3, ...],
                                            #   "WaterTankOff": [idx7, idx8, ...],
                                            #}




#ecnodes text labels into numberical ones using LabelEncoder() class
def encode_labels(global_label_encoder, labels):
    try:
        numeric_labels = global_label_encoder.transform(labels)
        return numeric_labels
    except ValueError as e:
        print(f"Error when encoding labels {e}")
        return np.array([])


#decodes numberical labels into text labels using LabelEncoder() class
def decode_labels(global_label_encoder, numeric_labels):
    # Die inverse_transform-Methode des LabelEncoders verwenden
    text_predictions = global_label_encoder.inverse_transform(numeric_labels)
    return text_predictions

def deduplicate_features(ds):
    """
    Removes duplicate rows from the feature matrix `ds`.
    Keeps only the first occurrence of each unique row.

    Parameters
    ----------
    ds : np.ndarray
        Feature matrix of shape (N, D)

    Returns
    -------
    ds_new : np.ndarray
        Deduplicated feature matrix.
    keep_indices : np.ndarray
        Indices of the rows that were kept (sorted, referring to original indexing).
    """
    ds = np.asarray(ds)

    # Unique rows, return_index gives index of FIRST occurrence
    _, keep_indices = np.unique(ds, axis=0, return_index=True)

    # Sort to preserve original ordering
    keep_indices = np.sort(keep_indices)

    ds_new = ds[keep_indices]

    print(
        f"[deduplicate_features] before: {len(ds)}, "
        f"after: {len(ds_new)}, "
        f"removed: {len(ds) - len(ds_new)} duplicates"
    )

    return ds_new


def deduplicate_labels_and_timestamps(labels, timestamps, keep_indices):
    """
    Applies deduplication to labels and timestamps using `keep_indices`
    obtained from deduplicating features.
    """
    labels = np.asarray(labels)
    timestamps = np.asarray(timestamps)
    keep_indices = np.asarray(keep_indices)

    labels_new = labels[keep_indices]
    timestamps_new = timestamps[keep_indices]

    print(f"[deduplicate_labels_and_timestamps] new length: {len(labels_new)}")

    return labels_new, timestamps_new


def deduplicate_folds(train_folds, test_folds, keep_indices):
    """
    Maps old fold indices (from the original dataset) into the new index
    space created after deduplication.

    Example:
        Old dataset has N samples.
        After deduplication only M samples remain.
        We need to map old indices -> new indices.
    """
    keep_indices = np.asarray(keep_indices)

    # Old index → new index mapping
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    #keep indices= [10,22,50]
    ## Resulting dictionary:
    # {10: 0, 22: 1, 50: 2}

    new_train_folds = []
    new_test_folds = []

    for tr, te in zip(train_folds, test_folds):
        tr_new = [index_map[i] for i in tr if i in index_map]
        te_new = [index_map[i] for i in te if i in index_map]

        new_train_folds.append(tr_new)
        new_test_folds.append(te_new)

    print("[deduplicate_folds] Fold remapping complete.")
    return new_train_folds, new_test_folds
