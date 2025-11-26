import argparse
import sys
from pathlib import Path

from cluster_based_on_candidates import create_and_analyze_clusters
from file_helper_t2 import read_df_from_csv, save_df_to_csv, \
    save_alignment_and_candidates_npz, load_alignment_and_candidates_npz
from process_pcap_t2 import preprocess_dataset
from unit_fields import build_fields_and_candidates_from_alignment
from sequence_alignment import start_sequence_alignment
from file_helper_t2 import create_large_csv_file_from_pcaps, read_df_from_csv
from unit_fields import build_fields_and_candidates_from_alignment
from sequence_alignment import start_sequence_alignment
from communication_sessions import group_into_communication_sessions_optimized, iat_gap_threshold
from communication_sessions import create_communication_sessions



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



def release_main():

    print("Executing functionality for reverse 1...")
    parser = argparse.ArgumentParser()
    args = sheet2_task1_args(parser, sys.argv[1:])
    release_main_reverse1(args)

    return


def release_main_reverse1(args):
    print(f"[reverse_1] Running task: {args.task}")

    #example: --task preprocess --dataset-dir "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set" --output-file "preprocess.csv"
    if args.task == "preprocess":
        # raw PCAPs -> preprocessed CSV
        preprocess_dataset(
            args.dataset_dir,
            args.output_file
        )

    # example: --task sessions --preprocessed-file "preprocessQUT.csv" --output-file "communicationSessions.csv"
    elif args.task == "sessions":
        # preprocessed CSV -> sessions CSV
        df=read_df_from_csv(args.preprocessed_file)
        df=create_communication_sessions(df)
        save_df_to_csv(df,args.output_file)

#--task align_keywords --preprocessed-file "preprocessQUT.csv"
    elif args.task == "align_keywords":
        # preprocessed CSV -> alignments + keyword candidates (combined file)
        df=read_df_from_csv(args.preprocessed_file)
        df_first_500 = df.head(500)

        print("Creating sequence alignment and keyword candidates for client traffic and saving it to 'client_alignment_and_candidates.npz'.")
        alignment_client, alignment_server = start_sequence_alignment(df_first_500)

        unit_fields_client, merged_fields_client, keyword_candidates_client = build_fields_and_candidates_from_alignment(
            alignment_client)
        save_alignment_and_candidates_npz("client_alignment_and_candidates.npz", alignment_client,
                                          keyword_candidates_client)

        print("Creating sequence alignment and keyword candidates for server traffic and saving it to 'server_alignment_and_candidates.npz'.")
        unit_fields_client, merged_fields_client, keyword_candidates_client = build_fields_and_candidates_from_alignment(
            alignment_client)
        save_alignment_and_candidates_npz("server_alignment_and_candidates.npz", alignment_client,
                                          keyword_candidates_client)

#example: #--task cluster_validate --preprocessed-file "preprocessQUT.csv"
    elif args.task == "cluster_validate":

        client_path = Path("client_alignment_and_candidates.npz")
        server_path = Path("server_alignment_and_candidates.npz")

        missing = []
        for p in [client_path, server_path]:
            if not p.exists():
                missing.append(p)

        if missing:
            print("ERROR: Required alignment/keyword files not found:")
            print("\nPlease run the 'align_keywords' task first to generate these files.")
            # terminate the program with non-zero exit code
            sys.exit(1)

        print("Cluster analysis for Client:\n\n")
        alignment_client_from_file, keyword_candidates_client= load_alignment_and_candidates_npz(
            "client_alignment_and_candidates.npz")
        create_and_analyze_clusters(alignment_client_from_file, keyword_candidates_client)

        print("Cluster analysis for Server:\n\n")
        alignment_server_from_file, keyword_candidates_server= load_alignment_and_candidates_npz(
            "server_alignment_and_candidates.npz")
        create_and_analyze_clusters(alignment_server_from_file, keyword_candidates_server)


    else:
        raise ValueError(f"Unknown task for reverse_1: {args.task}")
    return



def test_main():

    #call preprocessor instead
    #df=pcap_extract_values(args.dataset_dir)
    df=read_df_from_csv("/home/dW5kZWFk/uni/study_project/datasets/output/task2/preprocessQUT.csv")
    #print(len(df))

    df_first_100 = df.head(500)

    #group into client and server messages

    alignment_client, alignment_server=start_sequence_alignment(df_first_100)
    #print(len(alignment_server[0]))

    unit_fields_client, merged_fields_client, keyword_candidates_client = build_fields_and_candidates_from_alignment(alignment_client)

    create_and_analyze_clusters(alignment_client, keyword_candidates_client)
    print(keyword_candidates_client[1])

    #save_alignment_and_candidates_npz("client_alignment_and_candidates.npz", alignment_client, keyword_candidates_client)

    #unit_fields_server, merged_fields_server, keyword_candidates_server = build_fields_and_candidates_from_alignment(
    #    alignment_server)
    #save_alignment_and_candidates_npz("server_alignment_and_candidates.npz", alignment_server, keyword_candidates_server)
    return

    # load from file:
    # alignment_client_from_file, keyword_candidates_client_from_file=load_alignment_and_candidates_npz("client_alignment_and_candidates.npz"

    # print sequences
    # show_alignment_block_without_indices(alignment_client_from_file)

    # print keywords
    # for kc in keyword_candidates_client_from_file[:50]:
    #    print(kc)


if __name__ == "__main__":
    release_main()
