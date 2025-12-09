import argparse
import os
import sys
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder

from use_classifiers import execute_experiments_def, execute_experiments_abc
from handling_re_bytes_integrated import create_preprocessed_re_files
from k_fold import create_and_save_all_folds, save_folds_pretty
from use_classifiers import execute_scenario
from file_helper_t3 import verify_amount_feature_files,load_k_fold_results
from constants import ALL_POSSIBLE_LABELS
from feature_creation_autoencoder import train_and_save_models, create_features_for_ds_task3def
from feature_creation_autoencoder import create_features_for_ds
from preprocessing_s3t2 import  find_M, \
    pcaps_byte_and_metadata_extraction


def require_file(path: str):
    if not os.path.isfile(path):
        print(f"ERROR: Required file not found: {path}")
        print("Please run the the respective mode that creates this file first.")
        sys.exit(1)   # terminate program


def check_requirements_classifier_modes():

    require_file(f"k_fold_results/k_fold_s1_raw.json")
    training_indices_raw, test_indices_raw = load_k_fold_results(f"k_fold_results/k_fold_s1_raw.json")
    k = len(training_indices_raw)

    #check that k fold files for all scenarios exist
    for s_idx in range(1,4):
        require_file(f"k_fold_results/k_fold_s{s_idx}_raw.json")
        require_file(f"k_fold_results/k_fold_s{s_idx}_re.json")

        # check the amount of folds that was used when the training and test files were created
        training_indices_raw, test_indices_raw = load_k_fold_results(f"k_fold_results/k_fold_s{s_idx}_raw.json")
        training_indices_re, test_indices_re = load_k_fold_results(f"k_fold_results/k_fold_s{s_idx}_re.json")

        # feature files exist for all folds and number of folds should always be the same
        if len(training_indices_raw) != k or len(training_indices_re) != k :
            print("Run the k_fold mode with the same number for k first! Then run the extract_features mode with the same number of k.")
            sys.exit(1)

    if not verify_amount_feature_files(k):
        print("Run the k_fold mode with the same number for k first! Then run the extract_features mode with the same number of k.")
        sys.exit(1)

    #check that label and timestamp files exist
    require_file("datasets/raw_labels.npy")
    require_file("datasets/re_labels.npy")
    require_file("datasets/raw_timestamps.npy")
    require_file("datasets/re_timestamps.npy")
    return k


def check_requirements_feature_extraction_mode():
    k_folds_s1_raw_path = "datasets/k_fold_s1_raw.json"
    k_folds_s1_re_path = "datasets/k_fold_s1_raw.json"

    require_file(k_folds_s1_raw_path)
    require_file(k_folds_s1_re_path)

    require_file("k_fold_results/k_fold_s1_raw.json")
    require_file("k_fold_results/k_fold_s1_re.json")
    training_indices_raw, test_indices_raw = load_k_fold_results("k_fold_results/k_fold_s1_raw.json")
    training_indices_re, test_indices_re = load_k_fold_results("k_fold_results/k_fold_s1_re.json")

    k=len(training_indices_raw)

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
  3) extract_features       – train autoencoders and extract latent features
  4) classifiers            – run selected ML classifiers on chosen scenarios
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
    and stores the results to files.

    Prerequisites: 'dataset_preprocessing' was executed.

    K-Fold requirements:
      * Each data point appears EXACTLY ONCE in a test set across all k folds.
      * All folds are disjoint (no index appears in more than one fold).
      * The union of all folds contains every data point exactly once.
      * Fold sizes are as equal as possible; leftover samples are
        distributed across folds.
      * Duplicate samples in the dataset are allowed, but duplicate indices
        across folds are not.

  extract_features
    Uses training data from k-fold splits for Scenario 1 (control-only)
    to train autoencoders k times and then extract features for the
    entire dataset.

    It does not repeat this for Scenarios 2 and 3, because only control
    data is used for training and the control data per fold in the
    training set is the same for each scenario.

    Prerequisites:
      * 'dataset_preprocessing' was executed.
      * 'k_fold' was executed with the same k.

    Note:
      k autoencoders and feature files are created for each dataset
      (RAW and RE), with k taken from the k-fold setup.

  classifiers
    Trains and evaluates a selected Machine Learning algorithm using the
    k-fold results for a chosen scenario.

    Prerequisites:
      * 'dataset_preprocessing' executed
      * 'k_fold' executed
      * 'extract_features' executed

    Supported classifiers and scenarios:
      * One-Class SVM        – Scenario 1
      * Binary SVM           – Scenarios 2 and 3
      * Elliptic Envelope    – Scenario 1
      * Random Forest        – Scenarios 2 and 3
      * k-NN                 – Scenarios 2 and 3
      
    run_experiments_abc:
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on dataset RAW, suitable for task a to c.
    Will run with amount of folds created by previously executed k-folds mode. 
    
    run_experiments_def:
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on 3 diferent versions of RE dataset, preprocessed according to task d to f.
    Will run with amount of folds created by previously executed k-folds mode. 
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["dataset_preprocessing", "k_fold", "extract_features", "classifiers", "run_experiments_abc","run_experiments_def"],
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

    # Used in classifiers mode
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["ocsvm", "bsvm", "ee", "rf", "knn", "all"],
        help="Classifier to run when mode=classifiers.",
    )

    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        help=(
            "Scenario to run when mode=classifiers. "
            "If omitted, all valid scenarios for the chosen classifier are run."
        ),
    )

    return parser.parse_args()


# -------------------------------------------------------------------
# Mode implementations
# -------------------------------------------------------------------

def run_dataset_preprocessing(attack_dir: str, control_dir: str):
    """
    MODE: dataset_preprocessing

    Reads pcaps from attack_dir and control_dir, computes RAW and RE
    packet representations, and stores:
      * raw_bytes.npy / re_bytes.npy
      * raw_labels.npy / re_labels.npy
      * raw_timestamps.npy / re_timestamps.npy
    into the 'datasets' directory.
    """
    if not attack_dir or not control_dir:
        print("ERROR: --attack-dir and --control-dir are required for mode=dataset_preprocessing.")
        sys.exit(1)

    # If you still want to be able to auto-compute M_raw/M_re:
    # M_raw, M_re = find_M(attack_dir, control_dir)
    # For now, you fixed them:
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
    save_folds_pretty(k)

    print(f"[k_fold] Created and saved {k}-fold splits for all scenarios.")


def run_extract_features(k: int):
    """
    MODE: extract_features

    Trains autoencoders (for Scenario 1 control data) and extracts latent
    features for RAW and RE datasets using k folds.
    """
    check_requirements_feature_extraction_mode()
    os.makedirs("models", exist_ok=True)

    # Train k autoencoders (k can be deduced from training indices inside)
    train_and_save_models()

    # Use k when generating per-fold feature files
    create_features_for_ds(k)

    print("[extract_features] Trained autoencoders and extracted features.")


def run_classifiers(classifier: str | None,
                    scenario: int | None,
                    global_label_encoder: LabelEncoder):
    """
    MODE: classifiers

    Runs the selected classifier(s) on selected scenario(s).
    """
    # This function should:
    #   * verify that pre-steps are done
    #   * determine k from existing k-fold results
    k = check_requirements_classifier_modes()

    # If classifier is None, treat it as "all"
    if classifier is None:
        classifier = "all"

    # Helper to decide which scenarios are valid per classifier
    def valid_scenarios_for(clf_name: str):
        if clf_name in ("ocsvm", "ee"):
            return [1]
        elif clf_name in ("bsvm", "rf", "knn"):
            return [2, 3]
        else:
            # "all" -> all three scenarios
            return [1, 2, 3]

    # Helper to actually run one classifier on one scenario for both RAW/RE
    def run_one(clf_name: str, scen: int):
        print(f"\n[classifiers] Running {clf_name} on Scenario {scen} (RAW)\n")
        execute_scenario(
            global_label_encoder,
            classifier=clf_name,
            k=k,
            prefix="raw",
            scenario=scen,
        )

        print(f"\n[classifiers] Running {clf_name} on Scenario {scen} (RE)\n")
        execute_scenario(
            global_label_encoder,
            classifier=clf_name,
            k=k,
            prefix="re",
            scenario=scen,
        )

    # Determine which classifiers to run
    if classifier == "all":
        classifiers_to_run = ["ocsvm", "bsvm", "ee", "rf", "knn"]
    else:
        classifiers_to_run = [classifier]

    for clf_name in classifiers_to_run:
        scenarios_for_clf = valid_scenarios_for(clf_name)

        if scenario is not None:
            # User requested a specific scenario
            if scenario not in scenarios_for_clf:
                print(
                    f"WARNING: classifier '{clf_name}' is not defined for Scenario {scenario}. "
                    f"Valid scenarios for {clf_name}: {scenarios_for_clf}. Skipping."
                )
                continue
            # run only the requested scenario
            run_one(clf_name, scenario)
        else:
            # run all valid scenarios for this classifier
            for scen in scenarios_for_clf:
                run_one(clf_name, scen)


# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------




def release_main():
    args = parse_args()
    start = time.time()

    if args.mode == "dataset_preprocessing":
        run_dataset_preprocessing(args.attack_dir, args.control_dir)

    elif args.mode == "k_fold":
        run_k_fold(args.k)

    elif args.mode == "extract_features":
        run_extract_features(args.k)

    elif args.mode == "classifiers":
        # Build global label encoder once
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("models", exist_ok=True)
        run_classifiers(args.classifier, args.scenario, global_label_encoder)
    elif args.mode == "run_experiments_abc":
        #runs experiments a, b and c with grid search and saves results for precision and recall in files
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        k = check_requirements_classifier_modes()
        execute_experiments_abc(global_label_encoder, k) # if param is 0, so we use only the raw dataset. 
    elif args.mode == "run_experiments_def":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        #only re_bytes required, re_bytes5 etc will be created on the way
        require_file("datasets/re_bytes.npy")
        k = check_requirements_classifier_modes()
        create_preprocessed_re_files()

        #features are created yb re_byte5, re_byte10, re_byte15 etc...
        #with existing autoencoder (per fold)
        create_features_for_ds_task3def(k)  #needs server for RAM

        #1. integrate into classifier

        execute_experiments_def(global_label_encoder, k, "5")
        execute_experiments_def(global_label_encoder, k, "10")
        execute_experiments_def(global_label_encoder, k, "15")

        #2. remove duplicates directly before classification using "get_keep_indices_from_fold0"


    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    end = time.time()
    elapsed = end - start
    print(f"⏱️ release_main() executed in {elapsed:.2f} seconds")




def test_main():
    start = time.time()
    k=5

    re_bytes=np.load("datasets/re_bytes.npy")
    print(len(re_bytes))
    re_labels=np.load("datasets/re_labels.npy")

    re_labels = np.load("datasets/re_labels.npy")
    create_preprocessed_re_files()
    #create_preprocessed_re_files_with_seperation()
    re_bytes_15 = np.load("datasets/re_bytes_15.npy")
    print(len(re_bytes_15))
    re_bytes_10 = np.load("datasets/re_bytes_10.npy")
    print(len(re_bytes_10))
    re_bytes_5 = np.load("datasets/re_bytes_5.npy")
    print(len(re_bytes_5))


    return

    #create_features_for_ds_task3def(k)
    create_preprocessed_re_files()
    re_bytes_5=np.load("datasets/re_bytes_5/re5_features_fold0.npy")
    re_bytes_5 = np.load("datasets/re_bytes_5/re5_features_fold0.npy")
    re_bytes_5 = np.load("datasets/re_bytes_5/re5_features_fold0.npy")
    print(len(re_bytes_5))
    print(len(re_bytes_5[0]))

    re_bytes_10 = np.load("datasets/re_bytes_10/re10_features_fold0.npy")
    print(len(re_bytes_10))
    print(len(re_bytes_10[0]))

    re_bytes_15 = np.load("datasets/re_bytes_15.npy")
    print(len(re_bytes_15))
    print(len(re_bytes_10[0]))
    #create_preprocessed_re_files()

    #create_features_for_ds_task3def(k)

    end = time.time()  # end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")


if __name__ == "__main__":
#    test_main()
    release_main()
