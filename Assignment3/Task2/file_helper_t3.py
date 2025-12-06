import os
import pandas as pd
import json

#input: training_indices and test_indices for all folds [[],[],[],...] - k lists
def save_k_fold_results(training_indices, test_indices, filename):
    with open(filename, "w") as f:
        json.dump({
            "train_indices": training_indices,
            "test_indices": test_indices
        }, f)

    return


def load_k_fold_results(filename):

    with open(filename) as f:
        folds = json.load(f)

    train = folds["train_indices"]
    test  = folds["test_indices"]
    return train, test



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




