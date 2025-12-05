
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
from constants import ALL_POSSIBLE_LABELS
from Assignment3.Task2.constants import ATTACK_LABELS

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



#Label encoder



# Initialisieren Sie den Encoder nur einmal

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