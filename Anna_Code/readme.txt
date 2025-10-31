SERVER##############################################################################
PREPROCESS QUT ON THE SERVER:

python main.py --task preprocess --dataset qut --attack-dir /home/demboann/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks --control-dir /home/demboann/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --output-csv /home/demboann/datasets/output/2017QUT_S7Comm/preprocessQUT.csv


GET STATS FOR QUT ON SERVER:
/home/demboann/stats/2017QUT_S7Comm/

python main.py --task stats --dataset qut --stats-input-csv /home/demboann/datasets/output/2017QUT_S7Comm/preprocessQUT.csv --stats-output-dir /home/demboann/stats/2017QUT_S7Comm/


LOCAL#######################################################

python main.py --task preprocess --dataset qut --attack-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks --control-dir /home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set --output-csv /home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/preprocessQUT.csv

python main.py --task stats --dataset qut --stats-input-csv /home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/preprocessQUT.csv --stats-output-dir /home/dW5kZWFk/uni/study_project/stats/2017QUT_S7Comm

