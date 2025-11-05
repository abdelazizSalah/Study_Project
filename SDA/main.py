#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

from Anna_Code.file_helper import list_files_by_filetype
from SDA.autoencoder import hyperparameter_search, create_train_evaluate_model
from SDA.prepare_data import create_matrix_from_pcaps, make_datasets

from tensorflow import keras

from tensorflow.keras import layers
from evaluation_results import evaluate, extract_and_print_features_to_file



def create_all_models_extract_features(M, output_dir, ds_train, ds_test):

    #create 6 models: 2 different layers, 3 differnt activation function
    model_dense_relu, param_settings = hyperparameter_search(M, ds_train, ds_test, activation="relu", layer_type="dense")
    extract_and_print_features_to_file(ds_train, output_dir+"dense_relu", model_dense_relu, f"Model 1: Dense Layers, relu activation functions.\n Hyperparameters: {param_settings})")

    model_dense_elu, param_settings = hyperparameter_search(M, ds_train, ds_test, activation="elu", layer_type="dense")
    extract_and_print_features_to_file(ds_train, "dense_elu", model_dense_elu, f"Model 2: Dense Layers, elu activation functions.\n Hyperparameters: {param_settings})")


    model_dense_tanh, param_settings = hyperparameter_search(M, ds_train, ds_test, activation="tanh", layer_type="dense")
    extract_and_print_features_to_file(ds_train, "dense_tanh", model_dense_tanh, f"Model 3: Dense Layers, tanh activation functions.\n Hyperparameters: {param_settings})")


    model_conv1d_relu, param_settings = hyperparameter_search(M, ds_train, ds_test, activation="relu", layer_type="conv1d")
    extract_and_print_features_to_file(ds_train, "conv1d_relu", model_conv1d_relu, f"Model 3: conv1D Layers, relu activation functions.\n Hyperparameters: {param_settings})")

    model_conv1d_elu, param_settings = hyperparameter_search(M, ds_train, ds_test, activation="elu", layer_type="conv1d")
    extract_and_print_features_to_file(ds_train, "conv1d_elu", model_conv1d_elu, f"Model 3: conv1D Layers, elu activation functions.\n Hyperparameters: {param_settings})")

    model_conv1d_tanh, param_settings = hyperparameter_search(M, ds_train, ds_test, activation="tanh", layer_type="conv1d")
    extract_and_print_features_to_file(ds_train, "conv1d_tanh", model_conv1d_tanh, f"Model 3: conv1D Layers, tanh activation functions.\n Hyperparameters: {param_settings})")

    return


#/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set
def main():
    p = argparse.ArgumentParser(description="Minimal Autoencoder for ICS packets")
    p.add_argument("--pcap-dir", type=Path, help="Directory with PCAP/PCAPNG files (recursive)")
    p.add_argument("--M", type=int, required=True, help="Bytes per packet (input length)")
    p.add_argument("--epochs", required=True, type=int, help="")
    p.add_argument("--output-dir", required=True, type=Path, help="File to print features of best models.")

    args = p.parse_args()

    #create training data (np matrix)
    packet_length_M = args.M
    files=list_files_by_filetype(args.pcap_dir, "pcap")
    matrix=create_matrix_from_pcaps(files, packet_length_M)
    print(matrix.shape)

    #tansform np matrix into tensor dataset
    ds_train, ds_test, X_train, X_test = make_datasets(matrix)

    filename = args.output_file
    create_all_models_extract_features(filename, ds_train, ds_test)


#todo: transform simple AE into SDA
#todo: implement conv1d layer type
#todo: hyperparameter grid search (3b)
#todo: classifier (3d)




###################################################################################testing

def create_and_train_model(packet_length_M, path):

    files = list_files_by_filetype(path, "pcap")
    matrix = create_matrix_from_pcaps(files, packet_length_M)
    print(matrix.shape)

    # create training data (np matrix)
    ds_train, ds_test, X_train, X_test = make_datasets(matrix)

    #creat model
    param_settings=[]
    model_dense_relu, error = create_train_evaluate_model(packet_length_M, ds_train, ds_test, activation="relu", layer_type="dense", param_settings=param_settings)

    model_dense_relu.save("autoencoder_full_model_test.keras")


#--M 100 --epochs 100 --output-dir /home/dW5kZWFk/uni/study_project/stats/sda_output --pcap-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set
def main_for_tests():
    p = argparse.ArgumentParser(description="Minimal Autoencoder for ICS packets")
    p.add_argument("--pcap-dir", required=True, type=Path, help="Directory with PCAP/PCAPNG files (recursive)")
    p.add_argument("--M", type=int, required=True, help="Bytes per packet (input length)")
    p.add_argument("--epochs", required=True, type=int, help="")
    p.add_argument("--output-dir", required=True, type=Path, help="File to print features of best models.")

    args = p.parse_args()

    create_and_train_model(args.M, args.pcap_dir)

    #load_model_from_file
    model = keras.models.load_model("autoencoder_full_model_test.keras")
    print(model.summary())
    #test feature extraction print


if __name__ == "__main__":
    #main()
    main_for_tests()