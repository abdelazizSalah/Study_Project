import os
import pandas as pd
import numpy as np


def save_alignment_and_candidates_npz(filepath, aligned_messages, keyword_candidates):
    np.savez_compressed(
        filepath,
        aligned=np.array(aligned_messages, dtype=object),
        candidates=np.array(keyword_candidates, dtype=object)
    )
    print("Saved compressed:", filepath)


def load_alignment_and_candidates_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["aligned"].tolist(), data["candidates"].tolist()


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




