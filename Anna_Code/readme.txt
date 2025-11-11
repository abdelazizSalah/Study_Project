2. Usage
--------

    python main.py --task {preprocess|stats} --dataset {qut|electra} [options]


3. Tasks and Options
--------------------

### 3.1. Task: preprocess
Required for extracting stats.

#### A) QUT dataset
Combines ATTACK and CONTROL PCAPs into a single CSV file.

**Required arguments:**
    --attack-dir       Directory containing attack PCAPs
    --control-dir      Directory containing control (benign) PCAPs
    --output-file      Path to output CSV file

**Example:**
    python main.py --task preprocess --dataset qut --attack-dir /data/QUT/attacks --control-dir /data/QUT/control --output-file /data/processed/qut_2017.csv

#### B) Electra dataset
Converts a large CSV file into a compressed Parquet file.

**Required arguments:**
    --input-csv        Path to the Electra CSV file
    --output-file      Path to output Parquet file

**Example:**
    python main.py --task preprocess --dataset electra --input-csv /data/electra/electra_s7comm.csv --output-file /data/processed/electra.parquet


### 3.2. Task: stats
Generates statistics and plots from already preprocessed data (CSV or Parquet).

**Required arguments:**
    --stats-input-file   Path to preprocessed data file
    --stats-output-dir   Directory where statistics and images will be saved

#### A) QUT dataset
    python main.py --task stats --dataset qut --stats-input-file /data/processed/qut_2017.csv --stats-output-dir /data/stats/qut


#### B) Electra dataset
    python main.py --task stats --dataset electra --stats-input-file /data/processed/electra.parquet --stats-output-dir /data/stats/electra


Note: For 2017QUT_S7Comm all pcap.zip files have to be extracted before the execution


USAGE##############################################################################
(conda activate study-project)

QUTS7Comm - PREPROCESS ON SERVER:
python main.py --task preprocess --dataset qut --attack-dir /home/demboann/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks --control-dir /home/demboann/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --output-file /home/demboann/datasets/output/2017QUT_S7Comm/preprocessQUT.csv

QUTS7Comm - GET STATS ON SERVER:
python main.py --task stats --dataset qut --stats-input-file /home/demboann/datasets/output/2017QUT_S7Comm/preprocessQUT.csv --stats-output-dir /home/demboann/stats/2017QUT_S7Comm/

LOCAL#######################################################

python main.py --task preprocess --dataset qut --attack-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks --control-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --output-file /home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/preprocessQUT.csv

python main.py --task stats --dataset qut --stats-input-file /home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/preprocessQUT.csv --stats-output-dir /home/dW5kZWFk/uni/study_project/stats/2017QUT_S7Comm


ELECTRA - PREPROCESS ON SERVER:

python main.py --task preprocess --dataset electra --input-csv /home/demboann/datasets/electra_s7comm.csv  --output-file /home/demboann/datasets/output/electra/preprocessELECTRA.parquet

ELECTRA - GET STATS ON SERVER:

python main.py --task stats --dataset electra --stats-input-file /home/demboann/datasets/output/electra/preprocessELECTRA.parquet --stats-output-dir /home/demboann/stats/electra/
