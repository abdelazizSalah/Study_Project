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

from Anna_Code.cdf import print_cdf_task1d
from get_statistics import *
from process_pcap import *
from file_helper import *



def main():
    start = time.time()   # ⏱️ start timer


    #QUT S7Comm
    df=load_all_csvs()

    #Task01 A
    #print_packet_distribution_task1A(df)
    #print_packet_length_distribution_and_iat_task1B(df)
    #print_packet_distribution_task1C(df)

    #electra
    #df = dd.read_csv("/home/dW5kZWFk/Downloads/electra_s7comm.csv", blocksize="256MB")

    ## Check structure without reading full file
    #print(df.head())  # only reads the first partition
    #print(df.columns)  # metadata only
    #print(df.npartitions)  # how many chunks Dask created
    #col = df["Time"]
    #print(len(col))

    print_cdf_task1d(df)

    end = time.time()  # ⏹️ end timer
    elapsed = end - start
    print(f"⏱️ main() executed in {elapsed:.2f} seconds")
    return 0


main()