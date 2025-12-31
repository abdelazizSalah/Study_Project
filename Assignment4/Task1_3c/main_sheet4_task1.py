import argparse
import os
import sys
import time
from experiment_ae_classifier import run_experiment_ae_classifier
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sklearn.preprocessing import LabelEncoder
from error_overlap import all_error_overlaps
from measure_runtime import measure_all
from use_classifiers import execute_experiments_def, execute_experiments_abc
from handling_re_bytes_integrated import create_preprocessed_re_files
from k_fold import create_and_save_all_folds, print_k_fold_pretty
from file_helper_t3 import verify_amount_feature_files, load_k_fold_results
from constants import ALL_POSSIBLE_LABELS
from feature_creation_autoencoder import train_and_save_models, create_features_for_ds_task3def
from feature_creation_autoencoder import create_features_for_ds
from preprocessing_s3t2 import pcaps_byte_and_metadata_extraction
from feature_importance import all_feature_importance


def require_file(path: str):
    if not os.path.isfile(path):
        print(f"ERROR: Required file not found: {path}")
        print("Please run the the respective mode that creates this file first.")
        sys.exit(1)  # terminate program


def check_requirements_ae_classifier():
    require_file(f"k_fold_results/k_fold_s1_raw.json")
    training_indices_raw, test_indices_raw = load_k_fold_results(f"k_fold_results/k_fold_s1_raw.json")
    k = len(training_indices_raw)

    # check that k fold files for scenario 1 exists
    require_file(f"k_fold_results/k_fold_s1_raw.json")
    require_file(f"k_fold_results/k_fold_s1_re.json")

    # check the amount of folds that was used when the training and test files were created
    training_indices_raw, test_indices_raw = load_k_fold_results(f"k_fold_results/k_fold_s1_raw.json")
    training_indices_re, test_indices_re = load_k_fold_results(f"k_fold_results/k_fold_s1_re.json")

    # feature files exist for all folds and number of folds should always be the same
    if len(training_indices_raw) != k or len(training_indices_re) != k:
        print(
            "Run the k_fold mode with the same number for k first! Then run the extract_features mode with the same number of k.")
        sys.exit(1)



    # check that label and timestamp files exist
    require_file("datasets/raw_labels.npy")
    require_file("datasets/re_labels.npy")

    require_file("datasets/re_bytes.npy")
    require_file("datasets/re_bytes_5.npy")
    require_file("datasets/re_bytes_15.npy")
    require_file("datasets/re_bytes_10.npy")
    return k





def check_requirements_classifier_modes():
    require_file(f"k_fold_results/k_fold_s1_raw.json")
    training_indices_raw, test_indices_raw = load_k_fold_results(f"k_fold_results/k_fold_s1_raw.json")
    k = len(training_indices_raw)

    # check that k fold files for all scenarios exist
    for s_idx in range(1, 4):
        require_file(f"k_fold_results/k_fold_s{s_idx}_raw.json")
        require_file(f"k_fold_results/k_fold_s{s_idx}_re.json")

        # check the amount of folds that was used when the training and test files were created
        training_indices_raw, test_indices_raw = load_k_fold_results(f"k_fold_results/k_fold_s{s_idx}_raw.json")
        training_indices_re, test_indices_re = load_k_fold_results(f"k_fold_results/k_fold_s{s_idx}_re.json")

        # feature files exist for all folds and number of folds should always be the same
        if len(training_indices_raw) != k or len(training_indices_re) != k:
            print(
                "Run the k_fold mode with the same number for k first! Then run the extract_features mode with the same number of k.")
            sys.exit(1)

    if not verify_amount_feature_files(k):
        print(
            "Run the k_fold mode with the same number for k first! Then run the extract_features mode with the same number of k.")
        sys.exit(1)

    # check that label and timestamp files exist
    require_file("datasets/raw_labels.npy")
    require_file("datasets/re_labels.npy")
    return k


def check_requirements_feature_extraction_mode():
    require_file("k_fold_results/k_fold_s1_raw.json")
    require_file("k_fold_results/k_fold_s1_re.json")
    require_file("k_fold_results/k_fold_s2_raw.json")
    require_file("k_fold_results/k_fold_s2_re.json")
    require_file("k_fold_results/k_fold_s3_raw.json")
    require_file("k_fold_results/k_fold_s3_re.json")
    training_indices_raw, test_indices_raw = load_k_fold_results("k_fold_results/k_fold_s1_raw.json")

    k = len(training_indices_raw)

    return k


def check_requirements_k_fold_mode(k):
    if k > 8:
        print(
            "It is not possible to fullfill the requirements with k > 8 for the scenarios because one attack type has only 8 datapoints.")
        sys.exit(1)

    RAW_LABELS_PATH = "datasets/raw_labels.npy"
    RE_LABELS_PATH = "datasets/re_labels.npy"

    require_file(RAW_LABELS_PATH)
    require_file(RE_LABELS_PATH)
    return


def parse_args():
    description = """\
Study Project pipeline for S7Comm intrusion detection.

The tool supports several modes that correspond to the major pipeline steps:
  1) dataset_preprocessing  – read pcaps, build byte matrices, labels, timestamps (RAW and RE)
  2) k_fold                 – create k-fold splits for all scenarios
"""

    epilog = """\
MODE DETAILS

  dataset_preprocessing
    All operations are performed twice: once in RAW mode and once in RE mode.
    RAW: all bytes of each packet are included.
    RE: only the bytes after the keyword 'Candidate' are included
         (i.e., the physical S7Comm readings).

    For each mode (RAW and RE), the following steps are executed:
      0. Scan all pcaps to determine the maximum packet length
         (depending on RAW/RE).
      1. Read all pcaps and construct a byte matrix, where each row is
         one packet. Rows are padded or truncated to the mode-specific
         maximum length from step 0.
      2. Create a label list, where labels[i] is the label of bytes[i].
      3. Create a timestamp list, where timestamps[i] corresponds to
         bytes[i] and labels[i].

    Within each mode (RAW and RE separately), indices are aligned:
      bytes[i], labels[i], timestamps[i] all refer to the same packet.

  k_fold
    For each scenario (1–3) it creates k splits into training / test data
    and stores the results to files. It is performed once for RAW and once for RE.

    Prerequisites: 'dataset_preprocessing' was executed.

    K-Fold requirements:
      * Each data point appears EXACTLY ONCE in a test set across all k folds.
      * All folds are disjoint (no index appears in more than one fold).
      * The union of all folds contains every data point exactly once.
      * Fold sizes are as equal as possible; leftover samples are
        distributed across folds.
      * Duplicate samples in the dataset are allowed, but duplicate indices
        across folds are not.

  measure_runtime
     Prerequisites: 'dataset_preprocessing' and 'k_fold' was executed.

"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["dataset_preprocessing", "k_fold", "measure_runtime", "error_overlap", "feature_importance", "experiment_ae_classifier"],
        help="Which pipeline step to run.",
    )

    # Used in dataset_preprocessing mode
    parser.add_argument(
        "--attack-dir",
        type=str,
        help="Directory containing attack pcaps (required for mode=dataset_preprocessing).",
    )
    parser.add_argument(
        "--control-dir",
        type=str,
        help="Directory containing control pcaps (required for mode=dataset_preprocessing).",
    )

    # Used in k_fold and extract_features
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of folds to use for k-fold cross-validation (default: 5).",
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# Mode implementations
# -------------------------------------------------------------------

def run_dataset_preprocessing(attack_dir: str, control_dir: str):
    if not attack_dir or not control_dir:
        print("ERROR: --attack-dir and --control-dir are required for mode=dataset_preprocessing.")
        sys.exit(1)

    # M as length of longest packet per dataset
    # M_raw, M_re= find_M(attack_dir, control_dir)

    # to reduce runtime use fixed M, which was previously computed using the find_M function
    M_raw = 466
    M_re = 386

    os.makedirs("datasets", exist_ok=True)

    pcaps_byte_and_metadata_extraction(
        attack_dir,
        control_dir,
        "datasets/raw_bytes.npy",
        "datasets/re_bytes.npy",
        "datasets/raw_labels.npy",
        "datasets/re_labels.npy",
        "datasets/raw_timestamps.npy",
        "datasets/re_timestamps.npy",
        M_raw,
        M_re,
    )

    print("[dataset_preprocessing] Finished.")


def run_k_fold(k: int):
    """
    MODE: k_fold

    Creates k-fold splits for all scenarios and stores them in
    the 'k_fold_results' directory.
    """
    check_requirements_k_fold_mode(k)

    os.makedirs("k_fold_results", exist_ok=True)

    create_and_save_all_folds(k)

    print(f"[k_fold] Created and saved {k}-fold splits for all scenarios.")


def run_extract_features():
    """
    MODE: extract_features

    Trains autoencoders (for Scenario 1 control data) and extracts latent
    features for RAW and RE datasets using k folds training data .
    """
    k = check_requirements_feature_extraction_mode()
    os.makedirs("models", exist_ok=True)

    # Train k autoencoders
    train_and_save_models()

    # Use k when generating per-fold feature files
    create_features_for_ds(k)

    print("[extract_features] Trained autoencoders and extracted features.")



# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------


# from previous (sheet 3)
def release_main():
    args = parse_args()
    start = time.time()

    if args.mode == "dataset_preprocessing":
        run_dataset_preprocessing(args.attack_dir, args.control_dir)

    elif args.mode == "k_fold":
        run_k_fold(args.k)
        print_k_fold_pretty()
    elif args.mode == "extract_features":
        run_extract_features()  # needs server for RAM
    elif args.mode == "run_experiments_abc":
        # runs experiments a, b and c with grid search and saves results for precision and recall in files
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        k = check_requirements_classifier_modes()
        execute_experiments_abc(global_label_encoder, k)
    elif args.mode == "run_experiments_def":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        # only re_bytes required, re_bytes5 etc will be created on the way
        require_file("datasets/re_bytes.npy")  # additional requirement
        k = check_requirements_classifier_modes()
        create_preprocessed_re_files()  # creates re_bytes_5, re_bytes_10, re_bytes_15

        # features are created yb re_byte5, re_byte10, re_byte15 etc...
        # with existing autoencoder (per fold)
        create_features_for_ds_task3def(k)  # needs server for RAM

        execute_experiments_def(global_label_encoder, k, "5")
        execute_experiments_def(global_label_encoder, k, "10")
        execute_experiments_def(global_label_encoder, k, "15")



    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    end = time.time()
    elapsed = end - start
    print(f"⏱️ release_main() executed in {elapsed:.2f} seconds")


# for sheet 4 task1
def release_main_new():
    args = parse_args()

    if args.mode == "dataset_preprocessing":
        #run_dataset_preprocessing(args.attack_dir, args.control_dir)
        create_preprocessed_re_files()
    elif args.mode == "k_fold":
        run_k_fold(args.k)
        print_k_fold_pretty()
    elif args.mode == "measure_runtime":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_feature_extraction_mode()
        os.makedirs("results", exist_ok=True)
        measure_all(k, global_label_encoder)
    elif args.mode == "error_overlap":
        #execute measure runtime to generate feature files for raw and re15
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_feature_extraction_mode()
        all_error_overlaps(k, global_label_encoder)
    elif args.mode == "feature_importance":
        #execute measure runtime to generate feature files for raw and re15
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_feature_extraction_mode()
        all_feature_importance(k, global_label_encoder)
    elif args.mode == "experiment_ae_classifier":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        check_requirements_ae_classifier()
        run_experiment_ae_classifier(global_label_encoder)


if __name__ == "__main__":
    release_main_new()

