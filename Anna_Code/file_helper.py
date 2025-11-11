import os
import pandas as pd
from pathlib import Path

from process_pcap import pcap_extract_values


def save_df_as_parquet(df, path):
    df.to_parquet(path, compression="zstd", index=False)
    return 0


def read_df_from_parquet(path):
    df = pd.read_parquet(path, engine="pyarrow")
    return df


def save_df_to_csv(df,output_path, mode='w', header =True):
    df.to_csv(output_path, mode=mode, header=header, index=False)
    return 0


def read_df_from_csv(path):
    df=pd.read_csv(path, on_bad_lines="skip", engine="pyarrow") #pyarrow can read dfs faster
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

