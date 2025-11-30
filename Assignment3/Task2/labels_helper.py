
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



def get_attack_idx_by_type(labels):
    attack_idx_by_type = defaultdict(list)  #defaultdict: automatically creates a new list, when a new key is used

    for i, lab in enumerate(labels):
        if lab != "CONTROL":
            attack_idx_by_type[lab].append(i)

    return attack_idx_by_type               #{
                                            #   "ReactorOn": [idx1, idx2, idx3, ...],
                                            #   "WaterTankOff": [idx7, idx8, ...],
                                            #}