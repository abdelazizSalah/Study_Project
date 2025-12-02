import numpy as np
from sklearn.preprocessing import LabelEncoder

from Assignment3.Task2.constants import ALL_POSSIBLE_LABELS
from Assignment3.Task2.debugging_helpers import create_balanced_train_test_indices
from Assignment3.Task2.evaluation_validation import print_size_of_different_attack_types
from Assignment3.Task2.file_helper_t3 import save_k_fold_results, load_k_fold_results
from Assignment3.Task2.random_forest import execute_scenario_rf
from Assignment3.Task2.svm import execute_scenario_svm
from elliptic_envelope import execute_scenario_ee
from feature_creation import created_features_for_ds
from prepare_data_and_labels import pcaps_feature_and_attack_label_extraction
from tensorflow import keras
from k_fold import scenario1


def create_datasets_from_pcaps( input_path_pcap_attack, input_path_pcap_control):
    #creates file with raw bytes (length 100)
    #creates associated label files
    pcaps_feature_and_attack_label_extraction(input_path_pcap_attack, input_path_pcap_control, "datasets/raw_bytes.npy",
                                              "datasets/re_bytes.npy", "datasets/raw_labels.npy", "datasets/re_labels.npy",
                                              "datasets/raw_timestamps.npy","datasets/re_timestamps.npy")

    #based on those files creates files with features with autoencoder
    created_features_for_ds()
    return


def release_main():
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(ALL_POSSIBLE_LABELS)


    # preprocess with labels:
    """mode == dataset_preprocessing"""
    input_path_attack_pcap = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/20161216202830"
    input_path_control_pcap = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set"
    # toDo: output path
    create_datasets_from_pcaps(input_path_attack_pcap,input_path_control_pcap)

    # toDo timestamps
    """mode == k_fold"""
    #toDo: number of folds from commandline
    raw_labels=np.load("datasets/raw_labels.npy")
    training_indices, test_indices=scenario1(raw_labels,5)

    re_labels = np.load("datasets/raw_labels.npy")
    training_indices, test_indices = scenario1(re_labels, 5)


    #toDo: print to file for k=5, raw and re
    save_k_fold_results(training_indices, test_indices, "datasets/k_fold_s1_raw.json")


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
    input_path_pcap_attack = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/20161216202830"
    input_path_pcap_control = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set"
    pcaps_feature_and_attack_label_extraction(input_path_pcap_attack, input_path_pcap_control, "datasets/raw_bytes.npy",
                                              "datasets/re_bytes.npy", "datasets/raw_labels.npy",
                                              "datasets/re_labels.npy",
                                              "datasets/raw_timestamps.npy", "datasets/re_timestamps.npy")



if __name__ == "__main__":
    test_main()
    #extratestmainforlazypeople()