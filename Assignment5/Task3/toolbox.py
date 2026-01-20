'''
@Author: Abdelaziz Neamatallah
@Date: 18.11.25
@Desc: 
    # Create a tool box to combine all our piece of codes prepared in the previous sheet and this sheet. 
    # The tool should provide several functionalities that are read from the command line arguments
    # a. -h prints an overview information about the tool and list all implemented functionalities
    # b. -s <keyword> starts the toolbox in different operation modes
'''

import argparse
import sys, os
from pathlib import Path
def print_help():
    help_text = """
    Tool Box Overview:
    This tool provides several functionalities based on the operation mode specified.

    Available Functionalities:
        -h : Print this help overview and list all implemented functionalities.
        -s <keyword> : Start the toolbox in different operation modes based on the provided keyword.

    Possible Keywords for -s:
        reverse_1: performing sequence alignment and clustering the flows using first set of constraints.
        reverse_2: performing sequence alignment and clustering the flows using second set of constraints.
        statistics: compute and display various statistics from the dataset.
        similarity_flows: analyze and visualize flow similarities.
        sda: execution of unsupervised deep learning approach for anomaly detection using Stacked Denoising Autoencoders.

    For statistics:
       Required arguments: 
        --task {preprocess,stats} : Choose 'preprocess' or 'stats'.
        --dataset {qut,electra} : Dataset to handle.
         Preprocess arguments:
            QUT:
                --attack-dir : QUT preprocess: directory with ATTACK PCAPs
                --control-dir : QUT preprocess: directory with CONTROL PCAPs
                --output-file : Preprocess: path to output file (CSV for QUT, Parquet for Electra)
            Electra:
                --input-csv : Electra preprocess: path to the huge dataset CSV
                --output-file : Preprocess: path to output file.

        Statistics arguments:
            --stats-input-file : Stats: path to preprocessed data file 
            --stats-output-dir : Stats: directory to save results and images
    For SDA:
        Required arguments:
        --mode {models,features,classify} : Choose operation mode.
            models: train SDA models with hyperparameter search.
            features: export latent features for best models.
            classify: evaluate classifier using threshold y.
        Additional arguments may be required depending on the mode.
        # mode1: models
            --pcap-dir-control : Directory with PCAP/PCAPNG files for normal packets (recursive)
            --M : Bytes per packet (input length)
            --epochs : Training epochs
            
            Example:
            python toolbox.py -s sda --mode models --pcap-dir-control data/control_pcaps --M 100 --epochs 25

            This will:
            - Parse all .pcap files from data/control_pcaps/
            - Build and train 6 models (dense and conv1d, each with relu, elu, tanh)
            - Save trained models under models_and_data/
            - Save the generated dataset as models_and_data/dataset_control.npy
        # mode2: features
            --pcap-dir-control : Directory with PCAP/PCAPNG files for normal packets (recursive)
            --M : Bytes per packet (input length)
            --output-dir-features : Output directory for features of best SDA models.
            
            Example:
            python toolbox.py -s sda --mode features --pcap-dir-control data/control_pcaps --M 100 --output-dir-features data/features

            This will:
                - Load the trained models from models_and_data/
                - Load control dataset from models_and_data/dataset_control.npy (or create it, if it doesn't exist)
                - Extract Features for each model and save it into a file
                - Save each model’s features as .npy and .csv in output/features/
        # mode3: classify
            --pcap-dir-control : Directory with PCAP/PCAPNG files for normal packets (recursive)
            --pcap-dir-attack : Directory with PCAP/PCAPNG files for attack packets (recursive)
            --y : Threshold γ for MSE-based classifier.
            
            Example:
            python toolbox.py -s sda --mode classify --pcap-dir-control data/control_pcaps --pcap-dir-attack data/attack_pcaps --y 0.01

            This will:                
                - Ensure both dataset_control.npy and dataset_attack.npy exist (or create them if they do not exist)
                - Load the model models_and_data/model_dense_relu.keras (default)
                - Classify each packet as normal or attack based on reconstruction error:
                    avg(MSE) > γ  -> attack
                    otherwise     -> normal
                - Print counts of TP, FP, TN, FN and related metrics
    
    Example Usage:
    Help:
        python toolbox.py -h
    Similarity Flows Analysis:
        python toolbox.py -s similarity_flows
    Statistics Computation:
        QUT dataset: 
                python toolbox.py -s statistics --task stats --dataset qut --stats-input-file /data/processed/qut_2017.csv --stats-output-dir /data/stats/qut
        Electra dataset:
                python toolbox.py -s statistics --task stats --dataset electra --stats-input-file /data/processed/electra.parquet --stats-output-dir /data/stats/electra
    Reverse Engineering Task 1:
         python toolbox.py -s reverse_1 --task preprocess --dataset-dir ../../DataSets/2017QUT_S7comm/LabelledDataset --output-file ../../DataSets/output/task2/preprocessQUT.csv  

    Reverse Engineering Task 2:
        python toolbox.py -s reverse_2 (no args then defualt max_len = 4 is used)
        python toolbox.py -s reverse_2 -m 6 (max_len = 6 is used)
    
    GAN-based Anomaly Detection - Raw Packets:
        python toolbox.py -s GAN_raw --n 100 --mode D
        python toolbox.py -s GAN_raw --n 100 --mode G
    GAN-based Anomaly Detection - Reconstructed Packets:
        python toolbox.py -s GAN_re --n 100 --mode G
        python toolbox.py -s GAN_re --n 100 --mode D
    N-Gram Anomaly Detection:
        python toolbox.py -s ngram_detection --n {n-gram size}
    pipeline: run the Sheet 3–5 pipeline (dataset_preprocessing, k_fold, classifiers, cnn/resnet, ensemble, plots).
Example:
  python toolbox.py -s pipeline --h
  python toolbox.py -s pipeline --mode k_fold --k 5

    """
    print(help_text)
    

# I should merge Annas task1 arguments here too :)
def sheet1_task1_args(parser, remaining_args):

    parser.add_argument(
        "--task", required=True, choices=["preprocess", "stats"],
        help="Choose 'preprocess' or 'stats'."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["qut", "electra"],
        help="Dataset to handle."
    )

    # --- Preprocessing arguments ---
    parser.add_argument("--attack-dir", help="QUT preprocess: directory with ATTACK PCAPs")
    parser.add_argument("--control-dir", help="QUT preprocess: directory with CONTROL PCAPs")
    parser.add_argument("--input-csv", help="Electra preprocess: path to the huge dataset CSV")
    parser.add_argument("--output-file", help="Preprocess: path to output file (CSV for QUT, Parquet for Electra)")

    # --- Statistics arguments ---
    parser.add_argument("--stats-input-file", help="Stats: path to preprocessed file (CSV or Parquet) to analyze")
    parser.add_argument("--stats-output-dir", help="Stats: directory to save results and images")
    
    args = parser.parse_args(remaining_args)
    print(f"after parsing remaining args: {args}")

    print(f"TASK     = {args.task}")
    print(f"DATASET  = {args.dataset}")

    # print all arguments
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
    return args


def sheet1_sda_args(p, remaining_args):
    print(f'extracting args from task3 sheet1_sda_args with remaining args: {remaining_args}')
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

    args = p.parse_args(remaining_args)
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


def sheet2_task2_args(parser, remaining_args):
    
    # read the max_length from the arguments
    # define the argument parser

    # define the argument letter, its type, and its default value
    parser.add_argument('-m', '--max_len', type=int, default=4, help='Maximum length of keyword candidates to include')
    
    # read it from the command line
    args = parser.parse_args(remaining_args)

    return args

def sheet2_task1_args(parser, remaining_args):
    
    # Task and dataset selection
    parser.add_argument(
        "--task", required=True, choices=["preprocess", ""],
        help="Choose 'preprocess' or ''."
    )

    # --- Preprocessing arguments ---
    parser.add_argument("--dataset-dir", help="QUT preprocess: directory with all PCAPs in the dataset")
    parser.add_argument("--output-file", help="Preprocess: path to output file (CSV for QUT)")

    args = parser.parse_args(remaining_args)

    print(f"TASK     = {args.task}")
    return args


def sheet4_task2_raw_args(parser, remaining_args):
    """
    Argument parser for Sheet 4 - Task 2 (GAN-based anomaly detection)
    """

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of bytes per packet (input width n)"
    )

    parser.add_argument(
        "--mode",
        choices=["D", "G"],
        required=True,
        help="Inference mode: 'D' for Discriminator-based, 'G' for Generator-based"
    )

    args = parser.parse_args(remaining_args)

    print(f"N (bytes per packet) = {args.n}")
    print(f"MODE                = {args.mode}")

    return args

def sheet4_task3_ngram_args(parser, remaining_args):
    """
    Argument parser for Sheet 4 - Task 3 (N-gram based detection)
    """

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="N-gram size (e.g. 2, 3, 4)"
    )

    args = parser.parse_args(remaining_args)

    print(f"N-GRAM SIZE = {args.n}")

    return args






def load_all_modules():
    
    print(
                '''
        #############################################
        #-------------------------------------------#
        #       WELCOME TO AASP TOOL BOX  2025      #
        #   Electra & QUT S7comm Datasets will be   #
        #         easy for you to handle ;)         #
        #-------------------------------------------#
        #############################################

        '''
    )
    print('loading all necessary modules')
    
    # Adding assigment 1
    add_module_path('../../Assignment1/Abdelaziz_Codes/Sheet1_codes')
    add_module_path('../../Assignment1/Anna_Code')
    add_module_path('../../Assignment1/SDA')
    # Adding assigment 2
    add_module_path('../../Assignment2/Task1')
    add_module_path('../../Assignment2/Task2')

    # Adding assigment 3
    add_module_path('../../Assignment3/Task1')
    add_module_path('../../Assignment3/Task2')
    add_module_path('../../Assignment3/Task3')

    # Adding assigment 4 => done except Anna's part
    # add_module_path('../../Assignment4/Task1')
    add_module_path('../../Assignment4/Task2')
    add_module_path('../../Assignment4/Task3')
    # Adding assigment 5
    add_module_path('../../Assignment5/Task1')
    add_module_path('../../Assignment5/T1CNN_EC_ResNet')
    # add_module_path('../../Assignment5/Task2')

    
def add_module_path(module_path):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    modulte_abs_path = os.path.abspath(os.path.join(curr_dir, module_path))
    sys.path.insert(0,modulte_abs_path)
    # print(f'Added module path: {modulte_abs_path}')






def main():
    ##########################  Loading Modules #####################
    load_all_modules()
    from pathlib import Path
    from sheet1_task1 import release_main
    from task2_script import sheet1_task2
    from sheet1_task3 import sheet1_task3_main
    # from sheet2_task1 import release_main_reverse1
    from sheet2_task2 import sheet2_task2_reverse2
    from fine_tuning_task2_raw import task2_sheet4_main
    from task3_sheet4_ngrams_modifications import sheet4_task3_ngrams
    from fine_tuning_task2_re import task2_sheet4_main_finetuned_re
    from main_sheet5_task1CNN_EC import release_main_new


    ########################## Finish loading #####################


    parser = argparse.ArgumentParser(add_help=False, description="Tool Box for AASP 2025")
    parser.add_argument('-h', action='store_true', help='Print help overview')
    parser.add_argument('-s', type=str, help='Start toolbox in different operation modes')
    args, remaining_args = parser.parse_known_args()

    if args.s:
        print(f"Starting toolbox in mode: {args.s}")
        # Add code here to handle different operation modes based on args.s
        if args.s == 'similarity_flows':
            print("Executing flows functionality...")
            # Call relevant functions for mode 1
            sheet1_task2()  # Example call to a function from Sheet1_codes
        elif args.s == 'statistics':
            print(f"Executing {args.s} functionality...")
            parser = argparse.ArgumentParser()
            args = sheet1_task1_args(parser, remaining_args)
            release_main(args)
        elif args.s == 'sda':
            print("Executing functionality for mode 3...")
            parser = argparse.ArgumentParser()
            args = sheet1_sda_args(parser, remaining_args)
            # Call relevant functions for SDA with args
            sheet1_task3_main(args)

        elif args.s == 'reverse_1':
            print("Executing functionality for reverse 1...")
            parser = argparse.ArgumentParser()
            args = sheet2_task1_args(parser, remaining_args)
            release_main_reverse1(args)
            # Call relevant functions for mode 3
        elif args.s == 'reverse_2':
            print("Executing functionality for reverse 2...")
            parser = argparse.ArgumentParser()
            args = sheet2_task2_args(parser, remaining_args)
            sheet2_task2_reverse2(args)
            # Call relevant functions for mode 4
        elif args.s == 'GAN_raw':
            print("Executing functionality for reverse 2...")
            parser = argparse.ArgumentParser()
            args = sheet4_task2_raw_args(parser, remaining_args)
            task2_sheet4_main(args.n, args.mode)
        elif args.s == 'GAN_re':
            print("Executing functionality for reverse 2...")
            parser = argparse.ArgumentParser()
            args = sheet4_task2_raw_args(parser, remaining_args) # same args as raw
            task2_sheet4_main_finetuned_re(args.n, args.mode)
        elif args.s == 'ngram_detection':
            print("Executing functionality for reverse 2...")
            parser = argparse.ArgumentParser()
            args = sheet4_task3_ngram_args(parser, remaining_args)
            sheet4_task3_ngrams(args.n)
        elif args.s == 'pipeline':
            print("Executing Sheet3–5 pipeline...")
            # remaining_args already contains: --mode ... plus whatever else
            release_main_new(remaining_args)
        else:
            print(f"Mode {args.s} is not recognized.")
    elif args.h:
        print_help()
    else:
        print("No valid arguments provided. Use -h for help.")

if __name__ == "__main__":
    main()