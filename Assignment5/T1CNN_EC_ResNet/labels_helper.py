
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

    return attack_idx_by_type




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




def deduplicate_folds(train_folds, test_folds, keep_indices):
    """
    train_folds, test_folds: list of folds (each fold is a list of sample indices)
    keep_indices: list/array of sample indices to keep (original index space)

    Returns:
        new_train_folds, new_test_folds
        (same structure as input: list of folds, each fold is a list of indices)
    """

    keep = set(map(int, keep_indices))

    new_train_folds = []
    new_test_folds = []

    for tr, te in zip(train_folds, test_folds):
        tr_new = [i for i in tr if i in keep]
        te_new = [i for i in te if i in keep]
        new_train_folds.append(tr_new)
        new_test_folds.append(te_new)

    return new_train_folds, new_test_folds

