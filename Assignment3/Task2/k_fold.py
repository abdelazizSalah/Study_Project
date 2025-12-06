import os

import numpy as np
import random

from constants import ATTACK_LABELS
from file_helper_t3 import save_k_fold_results
from labels_helper import  split_attack_control, get_indices_for_attack_type


#input:  Fold (indices) = test data, number of samples in the whole dataset
#output:  training data (list with indices)
def get_train_indices_by_fold(all_indices, test_fold):
    #use set operations for efficiency
    all_indices_set = set(all_indices)
    current_fold_indices_set = set(test_fold)

    # difference between sets A\B (A without B)
    train_indices = list(all_indices_set - current_fold_indices_set)

    return train_indices    #they will be sorted!



#splits dataset into k folds
def create_folds_pure_python(data_indices, k):

    if k <= 1:
        raise ValueError("Choose K > 1")


    num_samples = len(data_indices) #number of samples (datapoints)

    #e.g. 1/5
    fold_size = num_samples // k

    # e.g. 12 % 5 = 2 more elements to be distributed among folds
    remainder = num_samples % k

    folds = []
    current_index = 0

    # create folds
    for i in range(k):

        if i<remainder: #compare index with remainder (e.g. 3 < 2)
            current_fold_size = fold_size + 1   # add an extra index (distribute remainder)
        else:
            current_fold_size = fold_size + 0

        # cut fold out of list with mixed indices
        fold = data_indices[current_index: current_index + current_fold_size]
        folds.append(fold)

        # update index start for next fold
        current_index += current_fold_size

    return folds    #[[],[],...] k lists inside


# k can't be > 8
def create_folds_for_all_attack_types(labels, k):
    """
    returns dictionary with k folds for each attack type
    attack_folds = {
        "A": [fold0, fold1, ..., fold_k-1],
        "B": [fold0, fold1, ..., fold_k-1],
        "C": [...],
        ...
    }"""

    attack_folds = {}

    for attack_label in ATTACK_LABELS:
        # collect indices for this attack type
        indices = [i for i, lab in enumerate(labels) if lab == attack_label]

        if len(indices) == 0:
            print(f"WARNING: No samples found for attack type '{attack_label}'")
            attack_folds[attack_label] = [[] for _ in range(k)] #add empty lists as value [[],[],...]
            continue

        # optional but shouldn't happen with k<9 restriction in main
        if len(indices) < k:
            raise ValueError(
                f"Attack type '{attack_label}' has only {len(indices)} samples, "
                f"but k={k}. Cannot ensure appearance in all folds."
            )

        # create the k folds
        folds = create_folds_pure_python(indices, k)

        attack_folds[attack_label] = folds  #[[],[],...]

    return attack_folds





def scenario1(labels, k):
    """
    -training data: only control
    -tests data (fold): control and attack (samples of each attack type)
    """

    control_indices, attack_indices = split_attack_control(labels)
    folds_only_control=create_folds_pure_python(control_indices,k)
    folds_by_attack_type=create_folds_for_all_attack_types(labels, k)

    training_indices = []  # [[],[]] sublist for each fold
    test_indices = []  # [[],[]] sublist for each fold

    for fold_idx in range(k):
        #create test set
        full_fold = list(folds_only_control[fold_idx])
        #add attack data
        for folds_by_type in folds_by_attack_type.values():
             full_fold+= folds_by_type[fold_idx]   #add attack indices to test fold

        # create training set
        training_indices_by_fold=get_train_indices_by_fold(control_indices, folds_only_control[fold_idx])
        training_indices.append(training_indices_by_fold)
        test_indices.append(full_fold)


    return training_indices, test_indices #for all folds [[],[],...]


def scenario2(labels, k):
    """
    -training data: control and n-2 attack types (left out attack types are chosen randomly for each fold)
    -tests data (fold): control and attack (samples of each attack type)
    """

    control_indices, attack_indices = split_attack_control(labels)
    folds_only_control=create_folds_pure_python(control_indices,k)
    folds_by_attack_type=create_folds_for_all_attack_types(labels, k)

    training_indices = []  # [[],[]] sublist for each fold
    test_indices = []  # [[],[]] sublist for each fold
    removed_types_per_fold = []

    attack_types = list(folds_by_attack_type.keys())


    for fold_idx in range(k):
        #remove two attack types for training data
        removed_types = random.sample(attack_types, 2)
        removed_types_per_fold.append(tuple(removed_types))

        #add control data
        full_fold = list(folds_only_control[fold_idx])   #test data only control
        full_training_for_fold = get_train_indices_by_fold(control_indices, full_fold) #rest of control data -> training set

        #add attack data
        for attack_type, folds_by_type in folds_by_attack_type.items():
             full_fold+= folds_by_type[fold_idx]   #add attack indices to test fold
             if attack_type not in removed_types:
                all_indices_for_attack_type=get_indices_for_attack_type(labels, attack_type)
                training_indices_by_fold_by_attack_type= get_train_indices_by_fold(all_indices_for_attack_type,
                                                                  folds_by_type[fold_idx])
                full_training_for_fold+=training_indices_by_fold_by_attack_type


        training_indices.append(full_training_for_fold)
        test_indices.append(full_fold)


    return training_indices, test_indices   #for all folds [[],[],...]



def scenario3(labels, k):
    """
    -training data: control and 1 attack type (chosen randomly for each fold)
    -tests data (fold): control and attack (samples of each attack type)
    """

    control_indices, attack_indices = split_attack_control(labels)
    folds_only_control=create_folds_pure_python(control_indices,k)
    folds_by_attack_type=create_folds_for_all_attack_types(labels, k)

    training_indices = []  # [[],[]] sublist for each fold
    test_indices = []  # [[],[]] sublist for each fold
    removed_types_per_fold = []

    attack_types = list(folds_by_attack_type.keys())


    for fold_idx in range(k):
        #choose one attack type for training data
        random_attack_type = random.choice(attack_types)

        #add control data to test and training set
        full_fold = list(folds_only_control[fold_idx])   #test data only control
        full_training_for_fold = get_train_indices_by_fold(control_indices, full_fold) #rest of control data -> training set

        #add attack data to test set
        for folds_by_type in folds_by_attack_type.values():
            full_fold += folds_by_type[fold_idx]  # add attack indices to test fold

        #add random attack type to training set
        all_indices_for_attack_type=get_indices_for_attack_type(labels, random_attack_type)
        training_indices_by_fold_by_attack_type= get_train_indices_by_fold(all_indices_for_attack_type,
                                                                  folds_by_attack_type[random_attack_type][fold_idx])
        full_training_for_fold+=training_indices_by_fold_by_attack_type


        training_indices.append(full_training_for_fold)
        test_indices.append(full_fold)


    return training_indices, test_indices   #for all folds [[],[],...]


def create_and_save_all_folds(k):

    labels_raw=np.load("datasets/raw_labels.npy")
    labels_re=np.load("datasets/re_labels.npy")

    scenario_numbers = [1, 2, 3]

    scenario_functions = {
        1: scenario1,
        2: scenario2,
        3: scenario3,
    }

    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs("k_fold_results", exist_ok=True)
    for s_num in scenario_numbers:
        scenario_func = scenario_functions[s_num]
        train_indices, test_indices = scenario_func(labels_raw,k)
        filename = f"k_fold_results/k_fold_s{s_num}_raw.json"
        save_k_fold_results(train_indices, test_indices, filename)
        print(f"Scenario {s_num} processed and results saved to {filename}")

    for s_num in scenario_numbers:
        scenario_func = scenario_functions[s_num]
        train_indices, test_indices = scenario_func(labels_re,k)
        filename = f"k_fold_results/k_fold_s{s_num}_re.json"
        save_k_fold_results(train_indices, test_indices, filename)
        print(f"Scenario {s_num} processed and results saved to {filename}")

    return 42


