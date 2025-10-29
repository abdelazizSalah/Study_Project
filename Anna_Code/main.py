# https://github.com/qut-infosec/2017QUT_S7comm/tree/master/LabelledDataset
#20161215163606_s7_process_attacks -> attack set
#20161219132813_control_set -> normal dataset

#How many network packets does each dataset contain in total?
# #How many of them are under attack and how many are normal packets?
# #How many and which application-layer network protocols are present in each dataset?
# #What is the fraction of packets exchanged though each of these protocols with respect to the total number of packets in each dataset;
#What is the fraction of packets under attack exchanged though each of these protocols?


import os
import time
from scapy.all import PcapReader, rdpcap
import pandas as pd
import dask.dataframe as dd

from Anna_Code.cdf import create_cdf_plots_task1d
from get_statistics import *
from process_pcap import *
from file_helper import *
from process_electra import read_electra


def main():
    start = time.time()   # ⏱️ start timer


    #QUT S7Comm
    #df=load_all_csvs("/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/all_attacks.csv",
                                   #  "/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/all_control.csv")

    #Task01 A
    #print_packet_distribution_task1A(df)
    #print_packet_length_distribution_and_iat_task1B(df)
    #print_packet_distribution_task1C(df)
    #create_cdf_plots_task1d(df)



    create_large_csv_file_from_pcaps("/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks",
                                     "/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set",
                                     "/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/all_attacks.csv",
                                     "/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/all_control.csv")



    #electra

    #read_electra("/home/dW5kZWFk/uni/study_project/datasets/Electra/electra_s7comm.csv")

    end = time.time()  # ⏹️ end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")
    return 0


main()