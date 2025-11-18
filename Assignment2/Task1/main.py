import argparse
import sys
from pathlib import Path
import time
import pandas as pd
from Assignment2.Task1.file_helper_t2 import create_large_csv_file_from_pcaps, read_df_from_csv
from sequence_alignment import iat_gap_threshold, find_threshold_iat_kmeans, group_into_communication_sessions, \
    group_into_communication_sessions_optimized


#--task preprocess --dataset-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset --output-file /home/dW5kZWFk/uni/study_project/datasets/output/task2/preprocessQUT.csv

def release_main():
    start = time.time()   # ⏱️ start timer

    parser = argparse.ArgumentParser(
        description="Argument handling for preprocessing or X (QUT)."
    )

    # Task and dataset selection
    parser.add_argument(
        "--task", required=True, choices=["preprocess", ""],
        help="Choose 'preprocess' or ''."
    )

    # --- Preprocessing arguments ---
    parser.add_argument("--dataset-dir", help="QUT preprocess: directory with all PCAPs in the dataset")
    parser.add_argument("--output-file", help="Preprocess: path to output file (CSV for QUT)")

    args = parser.parse_args()

    print(f"TASK     = {args.task}")

    if args.task == "preprocess":
        # validate args
        if not args.dataset_dir  or not args.output_file:
            sys.exit("Missing required args for QUT preprocess: --attack-dir --control-dir --output-file")

        dataset_dir = Path(args.dataset_dir)
        output_file = Path(args.output_file)

        if not dataset_dir.is_dir():
            sys.exit(f"--attack-dir not found or not a directory: {dataset_dir}")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print("MODE: QUT preprocess")
        print(f"attack_dir  = {dataset_dir}")
        print(f"output_file = {output_file}")

        create_large_csv_file_from_pcaps(str(dataset_dir), str(output_file))

    end = time.time()  # ⏹️ end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")



def test_main():
    df=read_df_from_csv("/home/dW5kZWFk/uni/study_project/datasets/output/task2/preprocessQUT.csv")



    threshold, small_cluster, large_cluster= iat_gap_threshold(df)
    print(threshold)
    print(len(small_cluster))
    print(len(large_cluster))
    #threshold: 0.1656239032745361

    unique_keys = df["session_id"].unique()
    print(unique_keys)
    print(len(unique_keys))
    df_grouped=group_into_communication_sessions_optimized(df, 0.1656)
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        print(df_grouped[["session_id", "iat_session_pair", "group_id"]].head(500))
    print(df_grouped[["session_id", "timestamp", "group_id"]].columns)


if __name__ == "__main__":
    test_main()
