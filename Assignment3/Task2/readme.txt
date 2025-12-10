README — Tasks 2 and 3

This project provides a complete pipeline for preprocessing network traffic, extracting features using an autoencoder, and evaluating several machine-learning models using k-fold validation. All functionality is executed through:

python main_sheet3_task2.py

For a full description of available modes, parameters, and usage examples, run:

python main_sheet3_task2.py -h

Installation:
Install all required dependencies using:

pip install -r requirements.txt

Execution Overview:
Tasks 2 and 3 are controlled using the --mode argument:

python main_sheet3_task2.py --mode <MODE> [additional parameters]

Each mode performs a specific part of the workflow. Some modes depend on files produced by earlier steps.

Mode Descriptions:

preprocess
Converts raw .pcap files into .csv files containing extracted bytes per packet.
Required for all subsequent modes.
This step loads packet captures, extracts per-packet byte sequences, and stores them in a consistent CSV format.

k-fold
Computes k-fold training/testing splits and stores them to disk.
Required for extract_features, classifiers, and experimental modes.
This step loads the preprocessed packets, splits the data into k folds depending on the scenario, and saves train/test index sets to files.

extract_features
Trains an autoencoder for each fold and extracts compressed feature representations.
Required for the classifiers mode.
This step loads the k-fold splits, trains an autoencoder on each fold’s training data, extracts latent features for training and test sets, and stores them for later use.

classifiers
Trains and evaluates the selected machine-learning model for a specified scenario.
This step loads the extracted features, trains the classifier for each fold, evaluates its performance, and prints metrics to the command line.
Supports multiple models (SVM, One-Class SVM, LOF, Random Forest, EllipticEnvelope, etc.).

run_experiments_abc
Runs predefined experiments on the RA dataset for Scenarios A, B, and C.

run_experiments_def
Runs predefined experiments on different RE datasets.

Workflow Summary:

preprocess

k-fold

extract_features

classifiers

run_experiments_abc or run_experiments_def (optional full pipeline experiments)

Help:
For detailed mode descriptions, parameter explanations, and usage examples, use:

python main_sheet3_task2.py -h