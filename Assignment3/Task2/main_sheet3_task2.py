import numpy as np


from Assignment3.Task2.evaluation_validation import print_amount_of_different_labels
from Assignment3.Task2.feature_creation import created_features_for_ds
from Assignment3.Task2.prepare_data_and_labels import pcaps_feature_and_attack_label_extraction
from tensorflow import keras



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

    return

def test_main():
    raw_bytes = np.load("datasets/raw_bytes.npy")
    #raw_labels=np.load("datasets/raw_labels.npy")
    raw_features=np.load("datasets/raw_features.npy")
    print(len(raw_features))
    re_bytes = np.load("datasets/re_bytes.npy")
    #re_labels = np.load("datasets/re_labels.npy")
    re_features = np.load("datasets/re_features.npy")
    print(len(re_features))
    #print_amount_of_different_labels(re_labels)

    #
    return


if __name__ == "__main__":
    test_main()