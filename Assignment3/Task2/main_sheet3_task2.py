import numpy as np


from Assignment3.Task2.evaluation_validation import print_size_of_different_attack_types
from Assignment3.Task2.file_helper_t3 import save_k_fold_results, load_k_fold_results
from feature_creation import created_features_for_ds
from prepare_data_and_labels import pcaps_feature_and_attack_label_extraction
from tensorflow import keras
from k_fold import start_folding, scenario1


def create_datasets_from_pcaps( input_path_pcap_attack, input_path_pcap_control):
    #creates file with raw bytes (length 100)
    #creates associated label files
    pcaps_feature_and_attack_label_extraction(input_path_pcap_attack, input_path_pcap_control, "datasets/raw_bytes.npy",
                                              "datasets/re_bytes.npy", "datasets/raw_labels.npy", "datasets/re_labels.npy")

    #based on those files creates files with features with autoencoder
    created_features_for_ds()
    return


def release_main():
    # preprocess with labels:
    """mode == dataset_preprocessing"""
    input_path_attack_pcap = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks/20161216202830"
    input_path_control_pcap = "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set"
    # toDo: output path
    create_datasets_from_pcaps(input_path_attack_pcap,input_path_control_pcap)

    """mode == k_fold"""
    #toDo: number of folds from commandline
    raw_labels=np.load("datasets/raw_labels.npy")
    training_indices, test_indices=scenario1(raw_labels,5)

    re_labels = np.load("datasets/raw_labels.npy")
    training_indices, test_indices = scenario1(re_labels, 5)


    #toDo: print to file for k=5, raw and re



    return

def test_main():
    raw_labels = np.load("datasets/raw_labels.npy")
    training_indices, test_indices = scenario1(raw_labels, 5)
    print(len(training_indices))
    print(len(test_indices))
    print(len(training_indices [2]))
    print(len(test_indices[3]))

    save_k_fold_results(training_indices, test_indices,"datasets/k_fold_s1.json")
    training_indices, test_indices=load_k_fold_results("datasets/k_fold_s1.json")

    print(len(training_indices))
    print(len(test_indices))
    print(len(training_indices[2]))
    print(len(test_indices[3]))

    re_labels = np.load("datasets/raw_labels.npy")
    training_indices, test_indices = scenario1(re_labels, 5)
    return



    raw_bytes = np.load("datasets/raw_bytes.npy")
    raw_labels=np.load("datasets/raw_labels.npy")
    raw_features=np.load("datasets/raw_features.npy")
    print(len(raw_features))
    re_bytes = np.load("datasets/re_bytes.npy")
    #re_labels = np.load("datasets/re_labels.npy")
    re_features = np.load("datasets/re_features.npy")
    print(len(re_features))
    #print_amount_of_different_labels(re_labels)

    start_folding(raw_bytes, raw_labels, k=5)



    return


if __name__ == "__main__":
    test_main()