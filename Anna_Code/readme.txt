Note: For 2017QUT_S7Comm all pcap.zip files have to be extracted before the execution


SERVER##############################################################################
conda activate study-project

PREPROCESS QUT ON THE SERVER:

python main.py --task preprocess --dataset qut --attack-dir /home/demboann/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks --control-dir /home/demboann/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --output-file /home/demboann/datasets/output/2017QUT_S7Comm/preprocessQUT.csv


GET STATS FOR QUT ON SERVER:
/home/demboann/stats/2017QUT_S7Comm/

python main.py --task stats --dataset qut --stats-input-file /home/demboann/datasets/output/2017QUT_S7Comm/preprocessQUT.csv --stats-output-dir /home/demboann/stats/2017QUT_S7Comm/



ELECTRA

PREPROCESS

python main.py --task preprocess --dataset electra --input-csv /home/demboann/datasets/electra_s7comm.csv  --output-file /home/demboann/datasets/output/electra/preprocessELECTRA.parquet

STATS

python main.py --task stats --dataset electra --stats-input-file /home/demboann/datasets/output/electra/preprocessELECTRA.parquet --stats-output-dir /home/demboann/stats/electra/



Run in Background on Server:
screen
LOCAL#######################################################

python main.py --task preprocess --dataset qut --attack-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks --control-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --output-file /home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/preprocessQUT.csv

python main.py --task stats --dataset qut --stats-input-file /home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/preprocessQUT.csv --stats-output-dir /home/dW5kZWFk/uni/study_project/stats/2017QUT_S7Comm

