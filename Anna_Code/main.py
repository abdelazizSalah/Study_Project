# https://github.com/qut-infosec/2017QUT_S7comm/tree/master/LabelledDataset
#20161215163606_s7_process_attacks -> attack set
#20161219132813_control_set -> normal dataset
import contextlib
#How many network packets does each dataset contain in total?
# #How many of them are under attack and how many are normal packets?
# #How many and which application-layer network protocols are present in each dataset?
# #What is the fraction of packets exchanged though each of these protocols with respect to the total number of packets in each dataset;
#What is the fraction of packets under attack exchanged though each of these protocols?


import os
import sys
import time
from scapy.all import PcapReader, rdpcap
import pandas as pd
import dask.dataframe as dd
import argparse

from toolz import excepts

from cdf import create_cdf_plots_task1d
from process_electra import read_electra_create_parquet
from get_statistics import *
from process_pcap import *
from file_helper import *
from process_electra import electra_df_extract_values


def test_main():
    start = time.time()   # start timer

    # create_large_csv_file_from_pcaps("/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks",
    #                                 "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set",
    #                                 "/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/qut_output.csv")
    #

    #QUT S7Comm
    df=read_df_from_csv("/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/all_attacks.csv")

    #Task01 A
    #print_packet_distribution_task1A(df)
    #print_packet_length_distribution_and_iat_task1B(df)
    #print_packet_distribution_task1C(df)
    #create_cdf_plots_task1d(df)


    #electra

    #create adapted csv out of electra
    read_electra_create_parquet("/home/dW5kZWFk/uni/study_project/datasets/Electra/electra_s7comm.csv", "/home/dW5kZWFk/uni/study_project/datasets/output/Electra/first_chunk.csv")

    #read csv into dataframe
    df=read_df_from_csv("/home/dW5kZWFk/uni/study_project/datasets/output/Electra/first_chunk.csv")
    #print(df)
    #print_packet_distribution_task1A(df)
    #print_packet_length_distribution_and_iat_task1B(df)
    print_packet_distribution_task1C(df)

    end = time.time()  #  end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")
    return 0


def save_statistics_to_file(df, output_dir, dataset_name):
    log_path = Path(output_dir) / f"{dataset_name}_statistics.txt"


#try:
    with open(log_path, "w") as f, contextlib.redirect_stdout(f):
        print(f"--- Statistics for dataset: {dataset_name} ---\n")

        print_packet_distribution_task1A(df)
        print_packet_length_distribution_and_iat_task1B(df)
        print_packet_distribution_task1C(df)

        print(f"\n--- Creating plots ---\n")

    create_cdf_plots_task1d(df, output_dir, dataset_name)

    print(f"Statistics successfully written to {log_path}")
#except Exception as e:
#    print(f"Error during retrieving statistics for {dataset_name}: {e}")


def release_main():
    start = time.time()   # ⏱️ start timer

    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Argument handling for preprocessing or statistics (QUT or Electra)."
    )

    # Task and dataset selection
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

    args = parser.parse_args()

    print(f"TASK     = {args.task}")
    print(f"DATASET  = {args.dataset}")

    if args.task == "preprocess":
        if args.dataset == "qut":
            # validate args
            if not args.attack_dir or not args.control_dir or not args.output_file:
                sys.exit("Missing required args for QUT preprocess: --attack-dir --control-dir --output-file")

            attack_dir = Path(args.attack_dir)
            control_dir = Path(args.control_dir)
            output_file = Path(args.output_file)

            if not attack_dir.is_dir():
                sys.exit(f"--attack-dir not found or not a directory: {attack_dir}")
            if not control_dir.is_dir():
                sys.exit(f"--control-dir not found or not a directory: {control_dir}")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            print("MODE: QUT preprocess")
            print(f"attack_dir  = {attack_dir}")
            print(f"control_dir = {control_dir}")
            print(f"output_file = {output_file}")

            create_large_csv_file_from_pcaps(str(attack_dir), str(control_dir), str(output_file))

        elif args.dataset == "electra":
            if not args.input_csv or not args.output_file:
                sys.exit("Missing required args for Electra preprocess: --input-csv --output-file")

            input_csv = Path(args.input_csv)
            output_file = Path(args.output_file)

            if not input_csv.is_file():
                sys.exit(f"--input-csv not found or not a file: {input_csv}")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            print("MODE: Electra preprocess")
            print(f"input_csv   = {input_csv}")
            print(f"output_file = {output_file}")

            read_electra_create_parquet(str(input_csv), str(output_file))

    elif args.task == "stats":
        if not args.stats_input_file or not args.stats_output_dir:
            sys.exit("Missing required args for stats: --stats-input-file --stats-output-dir")

        stats_input_file = Path(args.stats_input_file)
        stats_output_dir = Path(args.stats_output_dir)

        if not stats_input_file.is_file():
            sys.exit(f"--stats-input-file not found or not a file: {stats_input_file}")
        stats_output_dir.mkdir(parents=True, exist_ok=True)

        print("MODE: Statistics")
        print(f"stats_input_file = {stats_input_file}")
        print(f"stats_output_dir = {stats_output_dir}")


        if args.dataset == "qut":
            df = read_df_from_csv(str(stats_input_file))
            save_statistics_to_file(df, str(stats_output_dir), "2017QUT_S7Comm")
        elif args.dataset == "electra":
            df = read_df_from_parquet(str(stats_input_file))
            save_statistics_to_file(df, str(stats_output_dir), "Electra")
        else:
            sys.exit("Unknown dataset for stats (expected 'qut' or 'electra').")

    end = time.time()  # ⏹️ end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")

if __name__ == "__main__":
    release_main()
