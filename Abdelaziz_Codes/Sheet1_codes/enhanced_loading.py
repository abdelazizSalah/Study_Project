import pandas as pd
import sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append('../../Anna_Code')
from file_helper import *

def load_csv(file):
    # Use fast C parser and low_memory=False for better chunk merging
    return pd.read_csv(file, engine='c', low_memory=False)

def parallel_load(file_list, max_workers=None):
    dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_csv, f): f for f in file_list}
        for f in as_completed(futures):
            try:
                dfs.append(f.result())
                print(f'append succedded from {futures[f]}')
            except Exception as e:
                print(f"Error loading {futures[f]}: {e}")
    return dfs

if __name__ == "__main__":
    start = time.time()

    print('started processing')
    normal_files = sorted(list_files_by_filetype('../../DataSets/electra_s7comm/output/normal', 'csv'))
    attacked_files = sorted(list_files_by_filetype('../../DataSets/electra_s7comm/output/attacked', 'csv'))
    first_normal_df = load_csv(normal_files[0])
    print(first_normal_df.head())
    first_attacked_df = load_csv(attacked_files[0])
    print(first_attacked_df.head())

    print('starting parallelization')
    # Use half available cores to avoid memory contention
    import os
    max_workers = max(1, os.cpu_count() // 2)

    normal_dfs = [first_normal_df] + parallel_load(normal_files[1:], max_workers)
    attacked_dfs = [first_attacked_df] + parallel_load(attacked_files[1:], max_workers)
    print(f'end of parallelization: {time.time() - start}:.2f')

    # --- Combine normal dataframes ---
    output_normal = "../../DataSets/electra_s7comm/output/normal/combined_normal.csv"
    for i, df in enumerate(normal_dfs):
        mode = 'w' if i == 0 else 'a'          # write first, append others
        header = (i == 0)                      # header only once
        df.to_csv(output_normal, mode=mode, header=header, index=False)

    # --- Combine attacked dataframes ---
    output_attacked = "../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv"
    for i, df in enumerate(attacked_dfs):
        mode = 'w' if i == 0 else 'a' 
        header = (i == 0)
        df.to_csv(output_attacked, mode=mode, header=header, index=False)
    print('✅ All DataFrames appended successfully.')


    print(f"Loaded {len(normal_files)} normal and {len(attacked_files)} attacked files.")
    print(f"Total time: {time.time() - start:.2f} seconds")


# This scripts takes 26mins to be executed
