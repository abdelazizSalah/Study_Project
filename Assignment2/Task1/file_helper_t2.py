import os
import pandas as pd
from pathlib import Path

from process_pcap_t2 import pcap_extract_values

#
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
def create_large_csv_file_from_pcaps(input_path_pcap, output_csv_file):

    pcap_files=list_files_by_filetype(input_path_pcap,"pcap")

    first_control_file=1
    for path in pcap_files:

        filename = os.path.basename(path)
        try:
            write_header = first_control_file

            if first_control_file:
                file_mode = 'w'
                first_control_file=0
            else: file_mode='a'

            df_pcap=pcap_extract_values(path)
            save_df_to_csv(df_pcap, output_csv_file, mode=file_mode, header=write_header)
            print(f"File {path} done.")
        except Exception as e:
            print(f"Exception in {filename}:\n{e}")

    print("\n\nCreated CSV from PCAP files\n\n")


    return 0

