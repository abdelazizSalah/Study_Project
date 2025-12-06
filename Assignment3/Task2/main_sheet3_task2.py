import argparse
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder

from constants import ALL_POSSIBLE_LABELS
from debugging_helpers import create_balanced_train_test_indices
from feature_creation import train_and_save_models, create_features_for_ds
from file_helper_t3 import save_k_fold_results, load_k_fold_results
from random_forest import execute_scenario_rf
from svm import execute_scenario_svm
from elliptic_envelope import execute_scenario_ee
from feature_creation import create_features_for_ds, create_model
from prepare_data_and_labels import  find_M, \
    pcaps_byte_and_metadata_extraction
from k_fold import scenario1


def create_datasets_from_pcaps( input_path_pcap_attack, input_path_pcap_control):
    #creates file with raw bytes (length 100)
    #creates associated label files
    pcaps_byte_and_metadata_extraction(input_path_pcap_attack, input_path_pcap_control, "datasets/raw_bytes.npy",
                                              "datasets/re_bytes.npy", "datasets/raw_labels.npy", "datasets/re_labels.npy",
                                              "datasets/raw_timestamps.npy","datasets/re_timestamps.npy")

    #based on those files creates files with features with autoencoder
    created_features_for_ds()
    return


def require_file(path: str):
    if not os.path.isfile(path):
        print(f"ERROR: Required file not found: {path}")
        print("Please run the the respective mode that creates this file first.")
        sys.exit(1)   # terminate program


#todo: integrate
def parse_args():
    parser = argparse.ArgumentParser(description="Run scenario with user-selected number of folds.")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds to use for cross-validation (default: 5)"
    )
    return parser.parse_args()


def release_main():
    global_label_encoder = LabelEncoder() # use them to have the numeric labels first. 
    global_label_encoder.fit(ALL_POSSIBLE_LABELS)


    # preprocess with labels:
    """mode == dataset_preprocessing"""
    """
    All operations are performed twice: once in RAW mode and once in RE mode.
    RAW: all bytes of each packet are included.
    RE: only the bytes after the keyword Candidate are included (i.e., the physical S7Comm readings).

    For each mode, the following steps are executed:
        0 Scan all pcaps to determine the maximum packet length (i.e., the length of the longest RAW or RE packet, depending on the mode).
        1 Read all pcaps and construct a byte matrix, where each row represents one packet.
            All rows are padded or truncated to the mode-specific maximum length determined in Step 0.
        2 Create a label list, where each entry corresponds to the label of the packet in the same row of the byte matrix.
        3 Create a timestamp list, where each entry corresponds to the timestamp of the packet in the same row of the byte matrix.

        Within each mode (RAW and RE separately), the indices are aligned: bytes[i], labels[i], timestamps[i] all refer to the same packet. 
    """


    input_path_attack_pcap = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/20161216202830"
    input_path_control_pcap = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set"


    M_raw,M_re=find_M(input_path_attack_pcap,input_path_control_pcap)
    M_raw=466
    M_re=386 #todo: remove

    # exist_ok=True prevents an error if the directory already exists.
    os.makedirs("datasets", exist_ok=True)
    pcaps_byte_and_metadata_extraction(input_path_attack_pcap, input_path_control_pcap, "datasets/raw_bytes.npy",
                                              "datasets/re_bytes.npy", "datasets/raw_labels.npy",
                                              "datasets/re_labels.npy",
                                              "datasets/raw_timestamps.npy", "datasets/re_timestamps.npy", M_raw, M_re)


    """mode == k_fold"""
    """
    For each scenario (1 - 3) it creates k splits into training /test data. It stores the result to files.
    
    Prerequisites: Mode “Dataset_Preprocessing” was executed. 
    
    K-Fold requirements:
    Each data point must appear EXACTLY ONCE as part of a test set across all k folds. 
    All folds must be disjoint, meaning no index may appear in more than one fold. 
    The union of all folds must contain every data point exactly once. 
    Fold sizes should be as equal as possible, with leftover samples distributed across folds. 
    Duplicate samples in the dataset are allowed, but duplicate indices across folds are not. 
    """
    args = parse_args()
    k = args.k  #default k=5
    if k > 8:
        print("It is not possible to fullfill the requirements with k > 8 for the scenarios because one attack type has only 8 datapoints.")
        sys.exit(1)

    RAW_LABELS_PATH = "datasets/raw_labels.npy"
    RE_LABELS_PATH = "datasets/re_labels.npy"
    # Check required files
    require_file(RAW_LABELS_PATH)
    require_file(RE_LABELS_PATH)

    #toDo: number of folds from commandline
    raw_labels=np.load("datasets/raw_labels.npy")
    training_indices, test_indices=scenario1(raw_labels,k)

    re_labels = np.load("datasets/re_labels.npy")
    training_indices, test_indices = scenario1(re_labels, k)


    #toDo: print to file for k=5, raw and re
    save_k_fold_results(training_indices, test_indices, "datasets/k_fold_s1_raw.json")

    """mode extract_features"""
    """
    Uses training data from k-folds split for scenario 1 (which contains only control dps) to train autoencoder k times.
    
    Extracts Features for whole ds using the trained autoencoder for each split.
    (It does not repeat this for s2 and s3 because only control data can be used for training and
    the control data per fold inside the training set will be the same for each scenario.)
    
    Prerequisites: Mode “Dataset_Preprocessing” was executed and mode “K_Fold” was executed with the same number of folds (k). 
    """
    #extract features with autoencoder
    k_folds_s1_raw_path="datasets/k_fold_s1_raw.json"
    k_folds_s1_re_path = "datasets/k_fold_s1_raw.json"

    require_file(k_folds_s1_raw_path)
    require_file(k_folds_s1_re_path)
    args = parse_args()
    k = args.k

    training_indices_raw, test_indices_raw = load_k_fold_results("datasets/k_fold_s1_raw.json")
    training_indices_re, test_indices_re = load_k_fold_results("datasets/k_fold_s1_re.json")

    if len(training_indices_raw) != k or len(training_indices_re) != k :
        print("Run the k_fold mode with the same number for k first!")
        sys.exit(1)


    RAW_LABELS_PATH = "datasets/raw_labels.npy"
    RE_LABELS_PATH = "datasets/re_labels.npy"

    train_and_save_models()
    create_features_for_ds(k)

    """mode == OCSVM"""
    """svm s1 -> ocsvm"""
    training_indices, test_indices = load_k_fold_results("datasets/k_fold_s1_raw.json")  # one class in
    """raw"""
    raw_labels = np.load("datasets/raw_labels.npy")
    raw_features=np.load("datasets/raw_features.npy")
    raw_timestamps=np.load("datasets/raw_timestamps.npy", allow_pickle=True)

    execute_scenario_svm(raw_features, raw_labels, raw_timestamps, training_indices, test_indices, global_label_encoder, 1)

    """re"""
    re_labels = np.load("datasets/re_labels.npy")
    re_features = np.load("datasets/re_features.npy")
    re_timestamps = np.load("datasets/re_timestamps.npy", allow_pickle=True)
    training_indices, test_indices = load_k_fold_results("datasets/k_fold_s1_re.json")  # one class in

    execute_scenario_svm(re_features, re_labels, re_timestamps, training_indices, test_indices, global_label_encoder, 1)

    """mode == BSVM"""
    """svm s2 s3 -> svm"""
    # todo: load s2 or s3 indices, the following indices can be used for testing until task 2a is completed
    training_indices, test_indices = create_balanced_train_test_indices(raw_labels)
    training_indices = [training_indices]
    test_indices = [test_indices]
    """raw"""
    raw_labels = np.load("datasets/raw_labels.npy")
    raw_features = np.load("datasets/raw_features.npy")
    raw_timestamps = np.load("datasets/raw_timestamps.npy", allow_pickle=True)
    execute_scenario_svm(raw_features, raw_labels, raw_timestamps, training_indices, test_indices, global_label_encoder, 0)

    """re"""
    re_labels = np.load("datasets/re_labels.npy")
    re_features = np.load("datasets/re_features.npy")
    re_timestamps = np.load("datasets/re_timestamps.npy", allow_pickle=True)
    execute_scenario_svm(re_features, re_labels, re_timestamps, training_indices, test_indices, global_label_encoder, 0)

    """mode == Elliptic_Envelope"""
    """s1"""
    #load s1 indices
    training_indices, test_indices = load_k_fold_results("datasets/k_fold_s1_raw.json")  # one class in
    """raw"""
    raw_labels = np.load("datasets/raw_labels.npy")
    raw_features = np.load("datasets/raw_features.npy")
    raw_timestamps = np.load("datasets/raw_timestamps.npy", allow_pickle=True)
    execute_scenario_ee(raw_features, raw_labels, raw_timestamps, training_indices, test_indices, global_label_encoder)

    """re"""
    re_labels = np.load("datasets/re_labels.npy")
    re_features = np.load("datasets/re_features.npy")
    re_timestamps = np.load("datasets/re_timestamps.npy", allow_pickle=True)
    execute_scenario_ee(re_features, re_labels, re_timestamps, training_indices, test_indices, global_label_encoder)

    """mode=random forest"""
    """svm s2 s3 -> svm"""
    # todo: load s2 or s3 indices, the following indices can be used for testing until task 2a is completed
    training_indices, test_indices = create_balanced_train_test_indices(raw_labels)
    training_indices = [training_indices]
    test_indices = [test_indices]
    """raw"""
    raw_labels = np.load("datasets/raw_labels.npy")
    raw_features = np.load("datasets/raw_features.npy")
    raw_timestamps = np.load("datasets/raw_timestamps.npy", allow_pickle=True)
    execute_scenario_rf(raw_features, raw_labels, raw_timestamps, training_indices, test_indices, global_label_encoder)

    # -> s1: training data only has one class (control)
    return


def test_main():
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(ALL_POSSIBLE_LABELS)
    training_indices, test_indices = load_k_fold_results("datasets/k_fold_s1_raw.json")  # one class in

    raw_labels = np.load("datasets/raw_labels.npy")
    raw_features = np.load("datasets/raw_features.npy")
    raw_timestamps = np.load("datasets/raw_timestamps.npy", allow_pickle=True)
    #execute_scenario_ee(raw_features, raw_labels, raw_timestamps, training_indices, test_indices, global_label_encoder)

    # todo: load s2 or s3 indices, the following indices can be used for testing until task 2a is completed
    training_indices, test_indices = create_balanced_train_test_indices(raw_labels)
    training_indices = [training_indices]
    test_indices = [test_indices]
    execute_scenario_rf(raw_features, raw_labels, raw_timestamps, training_indices, test_indices, global_label_encoder)
    return
    #test binary classificators:
    training_indices, test_indices = create_balanced_train_test_indices(raw_labels)
    training_indices=[training_indices]
    test_indices=[test_indices]
    execute_scenario(raw_features,raw_labels,raw_timestamps, training_indices,test_indices, global_label_encoder, 0)

    return



def extratestmainforlazypeople():
    start = time.time()
    #train_and_save_models()
    create_features_for_ds(5)
    #features0=np.load("datasets/raw_features_fold0.npy")
    end = time.time()  # end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    #test_main()
    extratestmainforlazypeople()