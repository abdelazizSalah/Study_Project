import argparse
import os
import sys

from ensemble_classifier import run_experiment_ec
from use_classifiers import execute_experiments_abc, execute_experiments_def, execute_scenario
from feature_creation_autoencoder import  create_features_for_ds_raw, create_features_for_ds_re, \
    train_and_save_models_classifier
from measure_runtime import measure_all
from experiment_ae_classifier import run_experiment_ae_classifier, make_ae_metric_plots

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sklearn.preprocessing import LabelEncoder
from error_overlap import all_error_overlaps, plot_all_error_overlaps
from handling_re_bytes_integrated import create_preprocessed_re_files, get_keep_indices_from_fold0
from k_fold import create_and_save_all_folds, print_k_fold_pretty
from file_helper_t3 import load_k_fold_results, verify_amount_feature_files
from constants import ALL_POSSIBLE_LABELS
from preprocessing_s3t2 import pcaps_byte_and_metadata_extraction
from feature_importance import all_feature_importance, plot_all_feature_importance


def run_extract_features():
    """
    MODE: extract_features

    Trains autoencoders (for Scenario 1 control data) and extracts latent
    features for RAW and RE datasets using k folds training data .
    """
    k = check_requirements_feature_extraction_mode()
    os.makedirs("models", exist_ok=True)

    # Train k autoencoders
    prefixes=["raw","re5","re10","re15"]
    for prefix in prefixes:
        train_and_save_models_classifier(prefix)

    # Use k when generating per-fold feature files
    create_features_for_ds_raw(k)
    create_features_for_ds_re(k)
    print("[extract_features] Trained autoencoders and extracted features.")



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
    require_file("datasets/raw_timestamps.npy")
    require_file("datasets/re_timestamps.npy")
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


#Runs the selected classifier(s) on selected scenario(s).
def run_classifiers(classifier: str | None,
                    scenario: int | None,
                    global_label_encoder: LabelEncoder):

    k = check_requirements_classifier_modes()


    # Helper to decide which scenarios are valid per classifier
    def valid_scenarios_for(clf_name: str):
        if clf_name in ("ocsvm", "ee", "lof"):
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
        classifiers_to_run = ["ocsvm", "bsvm", "ee", "rf", "knn", "lof"]
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



def parse_args():
    description = """\
Study Project pipeline for S7Comm intrusion detection.

From Sheet 3 the 2 steps are always required:
  1) dataset_preprocessing  – read pcaps, build byte matrices, labels, timestamps (RAW and RE)
  2) k_fold                 – create k-fold splits for all scenarios
"""

    epilog = """\
MODE DETAILS

  dataset_preprocessing_sheet345
    This preprocessing mode only can be used for sheets 3,4 and 5.
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
  
  extract_features
    Uses training data from k-fold splits for Scenario 1 (control-only)
    to train autoencoders k times and then extract features for the
    entire dataset.

    It does not repeat this for Scenarios 2 and 3, because only control
    data is used for training and the control data per fold in the
    training set is the same for each scenario.

    Prerequisites:
      * 'dataset_preprocessing' was executed.
      * 'k_fold' was executed.

    Note:
      k autoencoders and feature files are created for each dataset
      (RAW and RE), with k taken from the k-fold setup.
  
  ASSIGNMENT 3 - Task 2:
  use_classifiers
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
      * Local Outlier Factor - Scenario 1
      
  ASSIGNMENT3 - Task 3:
  sheet3_run_experiments_abc:
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on dataset RAW, suitable for task a to c.
    Will run with amount of folds created by previously executed k-folds mode. 

  sheet3_run_experiments_def:
    Run all classifiers on all appropriate scenarios, collect per-fold
    precision/recall, and save them as CSV files in ./results.
    Runs on 3 different versions of RE dataset, preprocessed according to task d to f.
    Will run with amount of folds created by previously executed k-folds mode.     
  
  ASSIGNMENT 4 - Task 1:
  measure_runtime
     Measures average runtime and peak RAM for all classifiers using the datatypes RAW and RE15 over all folds.
     Creates plots.
     Prerequisites: 'dataset_preprocessing' and 'k_fold' was executed.

  error_overlap
    Measures error overlaps for all classifiers using the datatypes RAW and RE15 over all folds.
    Creates plots.
    Prerequisites: 'dataset_preprocessing', 'k_fold' and 'measure_runtime' was executed.

  feature_importance
    Measures feature importance for all classifiers using the datatypes RAW and RE15 over all folds. Creates plots.
    Prerequisites: 'dataset_preprocessing', 'k_fold' and 'measure_runtime' was executed.

  ASSIGNMENT 4 - Task 3c:
  experiment_ae_classifier
    Executes experiments on all datatypes and folds, using the autoencoder as a classifier. Evaluates results and creates plots.
    Prerequisites: 'dataset_preprocessing' and 'k_fold' was executed. Autoencoder base models for RAW and RE15 in the 'models' path.
"""

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["dataset_preprocessing_sheet345", "k_fold", "measure_runtime", "error_overlap", "feature_importance",
                 "experiment_ae_classifier", "extract_features", "use_classifiers", "sheet3_run_experiments_abc","sheet3_run_experiments_def", "ensemble_classifier" ],
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
        choices=["ocsvm", "bsvm", "ee", "rf", "knn", "lof"],
        help="Classifier to run when mode=classifiers.",
    )

    # Used in classifiers mode
    parser.add_argument(
        "--method",
        type=str,
        choices=["random", "majority", "all", "all_methods"], #all methods runs the previous 3
        help="Mode to run for ensemble classifier..",
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


# for sheet 4 task1
def release_main_new():
    args = parse_args()

    if args.mode == "dataset_preprocessing_sheet345":
        run_dataset_preprocessing(args.attack_dir, args.control_dir)
        create_preprocessed_re_files()
    elif args.mode == "k_fold":
        run_k_fold(args.k)
        print_k_fold_pretty()
    elif args.mode == "extract_features":
        run_extract_features()  # needs server for RAM
    #Assignment 3
    elif args.mode == "use_classifiers":
        if not args.classifier:
            print("ERROR: --classifier required for mode=use_classifiers")
        # Build global label encoder once
        # LabelEncoder from the Scikit-learn -> set up a way to convert text labels (categories) into numerical values for ML
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("models", exist_ok=True)
        k = check_requirements_classifier_modes()
        run_classifiers(args.classifier, args.scenario, global_label_encoder)
    elif args.mode == "sheet3_run_experiments_abc":
        # runs experiments a, b and c with grid search and saves results for precision and recall in files
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        k = check_requirements_classifier_modes()
        execute_experiments_abc(global_label_encoder, k)
    elif args.mode == "sheet3_run_experiments_def":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        os.makedirs("results", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        # only re_bytes required, re_bytes5 etc will be created on the way
        require_file("datasets/re_bytes.npy")  # additional requirement
        k = check_requirements_classifier_modes()

        execute_experiments_def(global_label_encoder, k, "5")
        execute_experiments_def(global_label_encoder, k, "10")
        execute_experiments_def(global_label_encoder, k, "15")

    #beginning Assignment 4
    elif args.mode == "measure_runtime":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_feature_extraction_mode()
        os.makedirs("results", exist_ok=True)
        measure_all(k, global_label_encoder)
    elif args.mode == "error_overlap":
        # execute measure runtime to generate feature files for raw and re15
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_feature_extraction_mode()
        all_error_overlaps(k, global_label_encoder)
        plot_all_error_overlaps()
    elif args.mode == "feature_importance":
        # execute measure runtime to generate feature files for raw and re15
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_feature_extraction_mode()
        all_feature_importance(k, global_label_encoder)
        plot_all_feature_importance()
    elif args.mode == "experiment_ae_classifier":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        check_requirements_ae_classifier()
        run_experiment_ae_classifier(global_label_encoder)
        make_ae_metric_plots()
    elif args.mode == "ensemble_classifier":
        global_label_encoder = LabelEncoder()
        global_label_encoder.fit(ALL_POSSIBLE_LABELS)
        k = check_requirements_classifier_modes()
        method=args.method
        run_experiment_ec(method,global_label_encoder, k)


def test_main():
    # keep_indices=get_keep_indices_from_fold0()
    param = 5
    keep_indices = get_keep_indices_from_fold0(f"datasets/re_bytes_{param}/", f"re{param}")



if __name__ == "__main__":
    release_main_new()
    #test_main()

