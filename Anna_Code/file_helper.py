import os
import pandas as pd
from pathlib import Path

from process_pcap import pcap_extract_values


def save_df_to_csv(df,output_path, mode, header ):
    df.to_csv(output_path, mode=mode, header=header, index=False)
    return 0


def read_df_from_csv(path):
    df=pd.read_csv(path, on_bad_lines="skip")
    return df


#list all files of certain filetype from directory and it's subdirectories
def list_files_by_filetype(root_path, filetype):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith("."+filetype):
                full_path = os.path.join(dirpath, filename)
                pcap_files.append(full_path)
    return pcap_files


#used for QUT_S7Comm
#one large file for attack dataset and control dataset
def create_large_csv_file_from_pcaps(input_path_attack_pcap,input_path_control_pcap,output_csv_file):
#assumes that control set is not empty, otherwise it will crash!

    pcap_files_attack=list_files_by_filetype(input_path_attack_pcap,"pcap")
    pcap_files_control=list_files_by_filetype(input_path_control_pcap,"pcap")

    first_control_file=1
    for path in pcap_files_control:

        filename = os.path.basename(path)
        try:
            write_header = first_control_file

            if first_control_file:
                file_mode = 'w'
                first_control_file=0
            else: file_mode='a'

            df_pcap=pcap_extract_values(path, 0)
            save_df_to_csv(df_pcap, output_csv_file, mode=file_mode, header=write_header)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")

    print("\n\nControl dataset done\n\n")

    first_attack_file = 1
    for path in pcap_files_attack:
        filename = os.path.basename(path)

        try:
            file_mode='a'
            df_pcap = pcap_extract_values(path, 1)
            save_df_to_csv(df_pcap, output_csv_file, mode=file_mode, header=0)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")


    return 0



###################################################### not used anymore:

#load all csv files from directory into one dataframe (or single csv file)
#output/2017QUT_S7comm/attacks/
#output/2017QUT_S7comm/control/
def load_all_csvs(attack_path, control_path):
    # Get all .csv files recursively (or non-recursively)
    if Path(attack_path).is_file():
        csv_files_attack = [str(attack_path)]
    else: csv_files_attack = list_files_by_filetype(attack_path,"csv")

    if Path(control_path).is_file():
        csv_files_control=[str(control_path)]
        print("yes")
    else: csv_files_control = list_files_by_filetype(control_path,"csv")
    csv_files=csv_files_attack+csv_files_control
    dfs = []


    #print(len(csv_files_attack))

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")


    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined.shape}")
    return combined



#used for QUT_S7Comm
#creates csv file for each PCAP file using the pcap_extract_values function
#(no csv created for empty files)
# 2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks
# 2017QUT_S7comm/LabelledDataset/20161219132813_control_set
# output/2017QUT_S7comm/attacks/
# output/2017QUT_S7comm/control/
def create_csv_file_for_each_pcap(input_path_attack_pcap,input_path_control_pcap,output_path_attack,output_path_control):

    pcap_files_attack=list_files_by_filetype(input_path_attack_pcap,"pcap")
    pcap_files_control=list_files_by_filetype(input_path_control_pcap,"pcap")


    for path in pcap_files_control:
        try:
            filename = os.path.basename(path)
            df_pcap=pcap_extract_values(path, 0)
            save_df_to_csv(df_pcap, output_path_control+filename+".csv", mode='w', header=1)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")

    print("\n\nControl dataset done\n\n")

    for path in pcap_files_attack:
        try:
            filename=os.path.basename(path)
            df_pcap = pcap_extract_values(path, 1)
            save_df_to_csv(df_pcap, output_path_attack+filename+".csv", mode='w', header=1)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {path}:\n{e}")
            os.path.getsize(path)

    return 0