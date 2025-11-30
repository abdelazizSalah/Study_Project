import numpy as np
import random

from Assignment3.Task2.constants import ATTACK_LABELS
from labels_helper import get_amount_of_different_attack_types, get_attack_idx_by_type, split_attack_control


#input:  Fold (indices) = test data, number of samples in the whole dataset
#output:  training data (list with indices)
def get_train_indices_by_fold(all_indices, test_fold):
    #use set operations for efficiency
    all_indices_set = set(all_indices)
    current_fold_indices_set = set(test_fold)

    # difference between sets A\B (A without B)
    train_indices = list(all_indices_set - current_fold_indices_set)

    return train_indices


#splits dataset into k folds
def create_folds_pure_python(data_indices, k):

    if k <= 1:
        raise ValueError("Choose K > 1")

    # mix
    random.seed(42) #same seed -> reproducable output for debugging todo: change to random
    random.shuffle(data_indices)

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


#x = 8*22
def choose_x_attack_indices_per_type(labels):
    counts={}
    attack_idx_by_type=get_attack_idx_by_type(labels)

    #get attack type with least amount of values -> to know the amount to collect fromm each type
    #8
    for lab, idx_list in attack_idx_by_type.items():
        length_of_indices = len(idx_list)
        counts[lab] = length_of_indices

    n_per_type = min(counts.values())  # 8

    rng = random.Random(42) #todo change to truly random

    chosen_attack_idx = []
    #iterate over each type
    for lab, idx_list in attack_idx_by_type.items():
        idx_list = list(idx_list)
        rng.shuffle(idx_list)
        chosen_attack_idx.extend(idx_list[:n_per_type]) #choose 8 indices from each type


    return  chosen_attack_idx #list with 8*n indices (n=amount of different attack types)

def scenario1(labels, k):
    """
    -training data: only control
    -tests data (fold): control and attack
        -> attack has x samples of each attack type (8, since x needs to represent all attack data)
    """
    #split dataset into attack and control data by labels
    control_indices, attack_indices=split_attack_control(labels)

    #create folds with only control data
    folds_only_control=create_folds_pure_python(control_indices,k)

    #get 8 attack indices for each type:
    attack_indices_selection=choose_x_attack_indices_per_type(labels)
    n_attack_indices=len(attack_indices_selection)    #176

    print(len(folds_only_control[0]))
    # check if fold size is smaller than 176
    if n_attack_indices>len(folds_only_control[0]):
        print("k too large")
        return

    training_indices=[] #[[],[]] sublist for each fold
    test_indices=[]    #[[],[]] sublist for each fold

    for control_fold in folds_only_control:
        full_fold = control_fold + attack_indices_selection   #add attack indices to test fold

        # also create trainings for each fold (only control ds as indices!)
        training_indices_by_fold=get_train_indices_by_fold(control_indices, control_fold)
        training_indices.append(training_indices_by_fold)
        test_indices.append(full_fold)

    return training_indices, test_indices


def start_folding(ds, labels, k):

    training_indices, test_indices=scenario1(labels,5)

    return
    print(ds[0])
    #create list of indices:
    num_samples = len(ds)

    all_indices = list(range(num_samples))
    folds=create_folds_pure_python(all_indices,k)
    print(folds[0][1])
    index_fold1=folds[0][1]
    print(labels[index_fold1])

    #train_indices=get_train_indices_for_fold(num_samples,folds[0])
    return