Three operation modes
via the --mode argument:

- models   : create and train models (with hyperparameter search)
- features : extract and save latent features from trained models
- classify : classify data and evaluate results using an MSE threshold γ

General Usage
-------------
python main.py --mode <mode> [other arguments...]

Mode 1 — models
---------------
Train and optimize all SDA variants using the control (normal) PCAP dataset.

Required arguments:
  --pcap-dir-control  Directory containing normal ICS traffic PCAP files
  --M                 Number of bytes per packet (input length)
  --epochs            Number of training epochs

Example:
  python main.py --mode models --pcap-dir-control data/control_pcaps --M 100 --epochs 25

This will:
  - Parse all .pcap files from data/control_pcaps/
  - Build and train 6 models (dense and conv1d, each with relu, elu, tanh)
  - Save trained models under models_and_data/
  - Save the generated dataset as models_and_data/dataset_control.npy


Mode 2 — features
-----------------
Extract latent (encoded) features from all previously trained models.

Required arguments:
  --pcap-dir-control     Directory containing the normal PCAPs (used if dataset missing)
  --M                    Input length (same as used during training)
  --output-dir-features  Directory where extracted features will be saved

Example:
  python main.py --mode features --pcap-dir-control data/control_pcaps --M 100 --output-dir-features output/features

This will:
  - Load all trained models from models_and_data/
  - Load control dataset from models_and_data/dataset_control.npy (or create it, if it doesn't exist)
  - Extract Features for each model and save it into a file
  - Save each model’s features as .npy and .csv in output/features/


Mode 3 — classify
-----------------
Run the SDA-based classifier to detect attacks using a threshold γ.

Required arguments:
  --pcap-dir-control  Directory with normal PCAPs (baseline)
  --pcap-dir-attack   Directory with attack PCAPs
  --y                 Threshold γ for reconstruction error

Example:
  python main.py --mode classify --pcap-dir-control data/control_pcaps --pcap-dir-attack data/attack_pcaps --M 100 --y 0.0001

This will:
  - Ensure both dataset_control.npy and dataset_attack.npy exist (or create them if they do not exist)
  - Load the model models_and_data/model_dense_relu.keras (default)
  - Classify each packet as normal or attack based on reconstruction error:
      avg(MSE) > γ  -> attack
      otherwise     -> normal
  - Print counts of TP, FP, TN, FN and related metrics
