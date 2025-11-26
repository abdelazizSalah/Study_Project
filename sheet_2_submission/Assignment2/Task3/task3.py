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
            python task3.py -s sda --mode models --pcap-dir-control data/control_pcaps --M 100 --epochs 25

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
            python task3.py -s sda --mode features --pcap-dir-control data/control_pcaps --M 100 --output-dir-features data/features

            This will:
                - Load the trained models from models_and_data/
                - Load control dataset from models_and_data/dataset_control.npy (or create it, if it doesn't exist)
                - Extract Features for each model and save it into a file
                - Save each model’s features as .npy and .csv in output/features/
        # mode3: classify
            --pcap-dir-control : Directory with PCAP/PCAPNG files for normal packets (recursive)
            --pcap-dir-attack : Directory with PCAP/PCAPNG files for attack packets (recursive)
            --y : Threshold for MSE-based classifier.
            
            Example:
            python task3.py -s sda --mode classify --pcap-dir-control data/control_pcaps --pcap-dir-attack data/attack_pcaps --y 0.01

            This will:                
                - Ensure both dataset_control.npy and dataset_attack.npy exist (or create them if they do not exist)
                - Load the model models_and_data/model_dense_relu.keras (default)
                - Classify each packet as normal or attack based on reconstruction error:
                    avg(MSE) >   -> attack
                    otherwise     -> normal
                - Print counts of TP, FP, TN, FN and related metrics
    
    Example Usage:
    Help:
        python task3.py -h
    Similarity Flows Analysis:
        python task3.py -s similarity_flows
    Statistics Computation:
        QUT dataset: 
                python task3.py -s statistics --task stats --dataset qut --stats-input-file /data/processed/qut_2017.csv --stats-output-dir /data/stats/qut
        Electra dataset:
                python task3.py -s statistics --task stats --dataset electra --stats-input-file /data/processed/electra.parquet --stats-output-dir /data/stats/electra
    Reverse Engineering Task 1:

        Argument parser for Sheet 2 - Reverse Engineering Task 1 (reverse_1).
        Modes (task):
            preprocess        : extract S7 values / sequences from raw PCAPs
            sessions          : create communication sessions (from preprocessed data)
            align_keywords    : sequence alignment and keyword candidate creation (from preprocessed data)
            cluster_validate  : clustering + cluster validation based on alignment/keywords (from alignemnt and keyword candidates)
    
        python task3.py -s reverse_1 --task preprocess --dataset-dir "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set" --output-file "preprocess.csv"
        python task3.py -s reverse_1 --task sessions --preprocessed-file "preprocessQUT.csv" --output-file "communicationSessions.csv"
        python task3.py -s reverse_1 --task align_keywords --preprocessed-file "preprocessQUT.csv"
        python task3.py -s reverse_1 --task cluster_validate --preprocessed-file "preprocessQUT.csv"

    Reverse Engineering Task 2:
        python task3.py -s reverse_2 (no args then defualt max_len = 4 is used)
        python task3.py -s reverse_2 -m 6 (max_len = 6 is used)
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

# def sheet2_task1_args(parser, remaining_args):
    
#     # Task and dataset selection
#     parser.add_argument(
#         "--task", required=True, choices=["preprocess", ""],
#         help="Choose 'preprocess' or ''."
#     )

#     # --- Preprocessing arguments ---
#     parser.add_argument("--dataset-dir", help="QUT preprocess: directory with all PCAPs in the dataset")
#     parser.add_argument("--output-file", help="Preprocess: path to output file (CSV for QUT)")

#     args = parser.parse_args(remaining_args)

#     print(f"TASK     = {args.task}")
#     return args



def sheet2_task1_args(parser, remaining_args):
    """
    Argument parser for Sheet 2 - Reverse Engineering Task 1 (reverse_1).
    Modes (task):
        preprocess        : extract S7 values / sequences from raw PCAPs
        sessions          : create communication sessions (from preprocessed data)
        align_keywords    : sequence alignment and keyword candidate creation (from preprocessed data)
        cluster_validate  : clustering + cluster validation based on alignment/keywords (from alignemnt and keyword candidates)
    """

    parser.add_argument(
        "--task",
        required=True,
        choices=["preprocess", "sessions", "align_keywords", "cluster_validate"],
        help="Operation mode for reverse_1."
    )

    parser.add_argument(
        "--dataset-dir",
        help="Directory with all PCAP/PCAPNG files of the dataset (used in 'preprocess')."
    )
    # preprocessed file (input for 'sessions' and 'align_keywords')
    parser.add_argument(
        "--preprocessed-file",
        help="Path to preprocessed packets file (used in 'sessions' and 'align_keywords')."
    )
    # generic output file:
    # - preprocess  -> preprocessed file
    # - sessions    -> sessions file
    # - cluster_validate -> clusters + validation results
    parser.add_argument(
        "--output-file",
        help="Output file path. For 'preprocess': preprocessed CSV; "
             "for 'sessions': sessions CSV; for 'cluster_validate': "
             "clusters + validation results."
    )

    args = parser.parse_args(remaining_args)
    # ---- check for presence of required arguments ----
    missing = []
    if args.task == "preprocess":
        if args.dataset_dir is None:
            missing.append("--dataset-dir")
        if args.output_file is None:
            missing.append("--output-file")
    elif args.task == "sessions":
        if args.preprocessed_file is None:
            missing.append("--preprocessed-file")
        if args.output_file is None:
            missing.append("--output-file")
    elif args.task == "align_keywords":
        if args.preprocessed_file is None:
            missing.append("--preprocessed-file")
    if missing:
        parser.error(f"Task '{args.task}' requires: {', '.join(missing)}")
    # debug print similar to sheet1_task1_args
    print(f"TASK (reverse_1) = {args.task}")
    for arg, value in vars(args).items():
        print(f"{arg:20s}: {value}")
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
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sheet1_codes_path = os.path.abspath(os.path.join(curr_dir, '..', '..','Assignment1','Abdelaziz_Codes' ,'Sheet1_codes'))
    Anna_codes_path = os.path.abspath(os.path.join(curr_dir,  '..', '..','Assignment1','Anna_Code'))
    sda = os.path.abspath(os.path.join(curr_dir,  '..', '..','Assignment1','SDA'))
    reverse2 = os.path.abspath(os.path.join(curr_dir,  '..','Task2'))
    reverse1 = os.path.abspath(os.path.join(curr_dir,  '..','Task1'))
    sys.path.append(sheet1_codes_path)
    sys.path.append(Anna_codes_path)
    sys.path.append(sda)
    sys.path.append(reverse2)
    sys.path.append(reverse1)
    



def main():
    ##########################  Loading Modules #####################
    load_all_modules()
    from sheet1_task1 import release_main
    from task2_script import sheet1_task2
    from sheet1_task3 import sheet1_task3_main
    from main_sheet2_task1 import release_main_reverse1
    from sheet2_task2 import sheet2_task2_reverse2
    ########################## Finish loading #####################


    parser = argparse.ArgumentParser(add_help=False, description="Tool Box for AASP 2025")
    parser.add_argument('-h', action='store_true', help='Print help overview')
    parser.add_argument('-s', type=str, help='Start toolbox in different operation modes')
    args, remaining_args = parser.parse_known_args()

    if args.h:
        print_help()
    elif args.s:
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
        else:
            print(f"Mode {args.s} is not recognized.")
    else:
        print("No valid arguments provided. Use -h for help.")

if __name__ == "__main__":
    main()