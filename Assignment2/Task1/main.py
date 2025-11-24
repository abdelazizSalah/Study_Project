import argparse
import sys
from pathlib import Path
import time
import pandas as pd
from Assignment2.Task1.file_helper_t2 import create_large_csv_file_from_pcaps, read_df_from_csv
from Assignment2.Task1.unit_fields import build_fields_and_candidates_from_alignment
from Assignment2.Task1.sequence_alignment import start_sequence_alignment
from communication_sessions import group_into_communication_sessions_optimized, iat_gap_threshold


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




def find_threshold():
    threshold, small_cluster, large_cluster= iat_gap_threshold(df)
    print(threshold)
    print(len(small_cluster))
    print(len(large_cluster))
    #threshold: 0.1656239032745361


def create_communication_groups(df):
    df_grouped=group_into_communication_sessions_optimized(df, 0.1656)
    with pd.option_context('display.max_columns', None, 'display.max_rows', None):
        print(df_grouped[["session_id", "iat_session_pair", "group_id"]].head(500))
    print(df_grouped[["session_id", "timestamp", "group_id"]].columns)
    return df_grouped



def test_main():
    df=read_df_from_csv("/home/dW5kZWFk/uni/study_project/datasets/output/task2/preprocessQUT.csv")
    #print(len(df))
    df_first_100 = df.head(500)

    #group into client and server messages

    alignment_client, alignment_server=start_sequence_alignment(df_first_100)
    #print(len(alignment_server[0]))

    unit_fields_client, merged_fields_client, keyword_candidates_client = build_fields_and_candidates_from_alignment(alignment_client)

    #save_alignment_and_candidates_npz("client_alignment_and_candidates.npz", alignment_client, keyword_candidates_client)

    unit_fields_server, merged_fields_server, keyword_candidates_server = build_fields_and_candidates_from_alignment(alignment_server)
    #save_alignment_and_candidates_npz("server_alignment_and_candidates.npz", alignment_server, keyword_candidates_server)


    #load from file:
    #alignment_client_from_file, keyword_candidates_client_from_file=load_alignment_and_candidates_npz("client_alignment_and_candidates.npz"

    #print sequences
    #show_alignment_block_without_indices(alignment_client_from_file)

    #print keywords
    #for kc in keyword_candidates_client_from_file[:50]:
    #    print(kc)

if __name__ == "__main__":
    test_main()
