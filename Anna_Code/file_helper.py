import os
import pandas as pd

from process_pcap import pcap_extract_values


def save_df_to_csv(df,output_path):
    df.to_csv(output_path, index=False)
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



#for each pcap file create a csv file using the pcap_extract_values function
#no csv created for empty files
def create_csv_files():

    pcap_files_attack=list_files_by_filetype("/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks","pcap")
    pcap_files_control=list_files_by_filetype("/home/dW5kZWFk/uni/study_project/datasets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set","pcap")

    output_base_path_attack="/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/attacks/"
    output_base_path_control = "/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/control/"
    print(len(pcap_files_attack))
    print(len(pcap_files_control))

    for path in pcap_files_control:
        try:
            filename = os.path.basename(path)
            df_pcap=pcap_extract_values(path, 0)
            save_df_to_csv(df_pcap, output_base_path_control+filename+".csv")
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")

    print("\n\nControl dataset done\n\n")

    for path in pcap_files_attack:
        try:
            filename=os.path.basename(path)
            df_pcap = pcap_extract_values(path, 1)
            #save_df_to_csv(df_pcap, output_base_path_attack+filename+".csv")
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {path}:\n{e}")
            os.path.getsize(path)

    return 0


#load all csv files into one dataframe
#used forQUT S7Comm
def load_all_csvs():
    # Get all .csv files recursively (or non-recursively)
    csv_files_attack = list_files_by_filetype("/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/attacks/","csv")
    csv_files_control = list_files_by_filetype("/home/dW5kZWFk/uni/study_project/datasets/output/2017QUT_S7comm/control/","csv")
    csv_files=csv_files_attack+csv_files_control
    dfs = []


    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Skipping {f}: {e}")


    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Combined DataFrame shape: {combined.shape}")
    return combined