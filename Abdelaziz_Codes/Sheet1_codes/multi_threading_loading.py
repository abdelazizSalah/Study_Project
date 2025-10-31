import pandas as pd
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append('../../Anna_Code')
from file_helper import *

def load_csv(file):
    return pd.read_csv(file)

start = time.time()

# list files
normal_files = list_files_by_filetype('../../DataSets/electra_s7comm/output/normal', 'csv')
attacked_files = list_files_by_filetype('../../DataSets/electra_s7comm/output/attacked', 'csv')

# parallel read
with ThreadPoolExecutor() as executor:
    normal_futures = {executor.submit(load_csv, f): f for f in normal_files}
    attacked_futures = {executor.submit(load_csv, f): f for f in attacked_files}

    normal_dfs = [f.result() for f in as_completed(normal_futures)]
    attacked_dfs = [f.result() for f in as_completed(attacked_futures)]

# combine
combined_normal_df = pd.concat(normal_dfs, ignore_index=True)
combined_attacked_df = pd.concat(attacked_dfs, ignore_index=True)
combined_normal_df.columns = ['Time', 'sip', 'dip','pair_ip','smac','dmac','pair_mac','packet_size', 'label','request']
combined_attacked_df.columns = ['Time', 'sip', 'dip','pair_ip', 'smac', 'dmac', 'pair_mac', 'packet_size', 'label','request']



print(f"Loaded {len(normal_files)} normal and {len(attacked_files)} attacked files.")
print(f"Total time: {time.time() - start:.2f} seconds")

'''
Loaded 388 normal and 388 attacked files.
Total time: 133.53 seconds1~
Loaded 388 normal and 388 attacked files.
Total time: 133.53 seconds
'''
