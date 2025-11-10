#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import numpy as np

from Anna_Code.file_helper import list_files_by_filetype
from SDA.autoencoder import hyperparameter_search, create_train_evaluate_model
from SDA.classifier import test_classifier
from SDA.prepare_data import create_matrix_from_pcaps, make_datasets, store_dataset_in_file

from evaluation_results import evaluate_model, extract_and_print_features_to_file
import tensorflow as tf
from tensorflow import keras




#run mode 2 "features" (requires existing models)
def print_features_for_all_models(output_dir):

    #load normal dataset from file
    data = np.load("models_and_data/dataset_normal.npy")

    ds_all = tf.data.Dataset.from_tensor_slices(data)
    ds_all_batched = ds_all.batch(128)

    #transform it into dataset (from npy format)

    model_dir = Path("models_and_data")
    model_names = [
        "model_dense_relu",
        "model_dense_elu",
        "model_dense_tanh",
        "model_conv1d_relu",
        "model_conv1d_elu",
        "model_conv1d_tanh",
    ]

    for model_name in model_names:
        try:
            model_path= model_dir / f"{model_name}.keras"
            trainedModel = keras.models.load_model(model_path)
            extract_and_print_features_to_file(ds_all_batched,  trainedModel, output_dir, model_name)
        except Exception as e:
            print(f"[!] Could not load model from {model_path}.\n Error: {e} \n "
                                "Make sure it exists and is a valid Keras model file (run in mode 'models' first.")

    return

from stackedDenoisingAutoEncoder import SDA 
#run mode 1 "model":
def create_and_train_all_optimized_models(M, ds_train,ds_validation, ds_test):
    '''
        The output here should be six trained models, each one with the best hyperparameters found.
        All the models should be saved to files for later use.

    '''
    layer_types = ["dense", "sparse"]
    activations = ["relu", "elu", "tanh"]
    numberOfLayers = [i for i in range(3,8)]  # from 3 to 7 layers
    optimizers = ['adam', 'gradient_descent']
    dropoutRates = [0.1, 0.2, 0.3, 0.4, 0.5]
    batchSizes = [64, 128, 256]
    noiseFactors = [0.1, 0.2, 0.3, 0.4, 0.5]
    epochs = [50, 100, 150]
    biasOptions = [True, False]
    #create 6 models: 2 different layers, 3 differnt activation function
    for layer_type in layer_types:
        for activation in activations:
            # for each model, we try to find the best hyperparameters
            bestModel = None
            bestMSE = float('inf')
            best_hyperparameters = {}   
            # All the followings are hyperparameters fine-tuning for each model type.
            for numLayer in numberOfLayers:
                for dropOut in dropoutRates: 
                    for batch in batchSizes: 
                        for noise in noiseFactors:
                            for epoch in epochs:
                                for optimizer in optimizers:
                                    for bias in biasOptions: 
                                        sda = SDA(
                                            numLayers= numLayer,
                                            hiddenNodesPerLayer = [M], # Ask how can we reduce this, if this is the output size? 
                                            dropoutPerLayer = [dropOut if layer_type == "dense" else 0.7],
                                            encodingActivationPerLayer = [activation],
                                            decodingActivationPerLayer = [activation],
                                            bias = bias,
                                            lossFunction = 'mse',
                                            batchSize = batch,
                                            numberOfEpochs = epoch,
                                            optimizer = optimizer, 
                                            noiseFactor = noise,
                                            layerType= layer_type,
                                            activationType= activation
                                        )

                                        # also the model is written in the given directory.
                                        print(f"\ncreating SDA with hyper parameters: {layer_type.upper()} model with activation {activation} ...")
                                        finalModel, trainingData, validationData, testingData, reconstructionMSE = sda.getSDAModel(ds_train, ds_validation, ds_test, 'models_and_data/')
                                        print(f"Model created and trained. Reconstruction MSE on test set: {reconstructionMSE}")
                                        if reconstructionMSE < bestMSE:
                                            bestMSE = reconstructionMSE
                                            bestModel = finalModel
                                            best_hyperparameters = {
                                                "layer_type": layer_type,
                                                "activation": activation,
                                                "num_layers": numLayer,
                                                "dropout": dropOut,
                                                "batch_size": batch,
                                                "noise_factor": noise,
                                                "epochs": epoch,
                                                "optimizer": optimizer,
                                                "bias": bias
                                            }

    return bestModel, best_hyperparameters



def parse_args():
    p = argparse.ArgumentParser(description="Stacked Denoising Autoencoder for ICS packets")

    # choose behavior
    p.add_argument(
        "--mode",
        choices=["models", "features", "classify"],
        required=True,
        help=(
            "Mode 'models': train SDA models with hyperparameter search. "
            "Mode 'features': export latent features for best models. "
            "Mode 'classify': evaluate classifier using threshold y."
        ),
    )

    # shared / potentially used in multiple modes
    p.add_argument("--pcap-dir-control", type=Path,
                   help="Directory with PCAP/PCAPNG files for normal packets (recursive)")
    p.add_argument("--M", type=int, help="Bytes per packet (input length)")
    p.add_argument("--epochs", type=int, help="Training epochs (used in 'models', possibly 'features')")

    # mode 2 only
    p.add_argument("--output-dir-features", type=Path,
                   help="Output directory for features of best SDA models (mode 'features').")

    # mode 3 only
    p.add_argument("--pcap-dir-attack", type=Path,
                   help="Directory with PCAP/PCAPNG files for attack packets (mode 'classify').")
    p.add_argument("--y", type=float,
                   help="Threshold γ for MSE-based classifier (mode 'classify').")

    args = p.parse_args()

    # --- conditional requirements ---

    if args.mode == "models":
        missing = []
        if args.pcap_dir_control is None:
            missing.append("--pcap-dir-control")
        if args.M is None:
            missing.append("--M")
        if args.epochs is None:
            missing.append("--epochs")
        if missing:
            p.error(f"Mode 'models' requires: {', '.join(missing)}")

    elif args.mode == "features":
        # needs normal data + M + where to put features
        missing = []
        if args.pcap_dir_control is None:
            missing.append("--pcap-dir-control")
        if args.M is None:
            missing.append("--M")
        if args.output_dir_features is None:
            missing.append("--output-dir-features")
        if missing:
            p.error(f"Mode 'features' requires: {', '.join(missing)}")

    elif args.mode == "classify":
        # needs normal (to train/load model or baseline), threshold y
        missing = []
        if args.pcap_dir_control is None:
            missing.append("--pcap-dir-control")
        if args.pcap_dir_attack is None:
            missing.append("--pcap-dir-attack")
        if args.y is None:
            missing.append("--y")
        if missing:
            p.error(f"Mode 'classify' requires: {', '.join(missing)}")

    return args


def main():
    start = time.time()   #  start timer

    args = parse_args()

    normal_path = Path("models_and_data/dataset_normal.npy")
    attack_path = Path("models_and_data/dataset_attack.npy")
    have_normal = normal_path.exists()
    have_attack = attack_path.exists()

    #verify that normal dataset already exists in file (required for mode features and mode classify)
    if not have_normal:
        print(
            f"No dataset file found for normal packets ('models_and_data/dataset_normal.npy')! The dataset will be created from the pcap files in{args.pcap_dir_attack}. This may take a while...")
        store_dataset_in_file(args.pcap_dir_control, args.M, 0)
        

    #for creation of models, dataset will be prepared
    if args.mode == "models":
        print("Create Models Mode\n")
        # load normal dataset from file
        data = np.load("models_and_data/dataset_normal.npy")

        # convert it to training, validation and test sets
        ds_training = data[:int(0.8 * data.shape[0])] # 80% training
        ds_validation = data[int(0.8 * data.shape[0]):int(0.9 * data.shape[0])] # 10% validation
        ds_test = data[int(0.9 * data.shape[0]):] # 10% test
        create_and_train_all_optimized_models(args.M, ds_training,ds_validation, ds_test) #stores models in files

    #--mode features --pcap-dir-control /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --M 100 --output-dir-features /home/dW5kZWFk/uni/study_project/stats/sda_output
    elif args.mode == "features":
        print("Print features Mode\n")
        print_features_for_all_models(args.output_dir_features)

    #--mode classify --pcap-dir-control /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set
    #--pcap-dir-attack /home/dW5kZWFk/Desktop/test --y 0.0001
    elif args.mode == "classify":
        print("Test classifier Mode\n")
        if (not have_attack):
            print(
                f"No dataset file found for attack packets ('models_and_data/dataset_attack.npy')! The dataset will be created from the pcap files in{args.pcap_dir_attack}. This may take a while...")
            store_dataset_in_file(args.pcap_dir_attack, args.M, 1)

        model_name="models_and_data/model_dense_relu.keras"
        trainedModel = keras.models.load_model(model_name) #choose model that should be tested!
        print(f"Classifier will be tested for {model_name} with threshold {args.y}.")
        test_classifier(trainedModel, y=args.y)

    end = time.time()  # end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")

#todo: transform simple AE into SDA
#todo: implement conv1d layer type
#todo: hyperparameter grid search (3b)
#todo: classifier (3d)

#--M 100 --epochs 100 --output-dir /home/dW5kZWFk/uni/study_project/stats/sda_output --pcap-dir-control /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set
#--pcap-dir-attack /home/dW5kZWFk/Desktop/test
def main_for_tests():
    p = argparse.ArgumentParser(description="Minimal Autoencoder for ICS packets")
    p.add_argument("--pcap-dir-control", required=True, type=Path, help="Directory with PCAP/PCAPNG files for normal packets (recursive)")
    p.add_argument("--M", type=int, required=True, help="Bytes per packet (input length)")
    p.add_argument("--epochs", required=True, type=int, help="")
    p.add_argument("--output-dir", required=True, type=Path, help="Directory to print features of best SDA models.")
    p.add_argument("--pcap-dir-attack",  required=True, type=Path, help="Directory with PCAP/PCAPNG files for attack packets. Required for Classification!")
    p.add_argument("--y",  required=True, type=float, help="Threshold for MSE - used for classifier.")


    args = p.parse_args()

    trainedModel = keras.models.load_model("models_and_data/model_dense_relu.keras")
    test_classifier(trainedModel, y=args.y)

    return

if __name__ == "__main__":
    main()
