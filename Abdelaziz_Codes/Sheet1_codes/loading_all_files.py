# read all CSV files 
import sys
import time
sys.path.append('../../Anna_Code')  
from file_helper import *

start = time.time()
normal_files = list_files_by_filetype('../../DataSets/electra_s7comm/output/normal','csv')
attacked_files = list_files_by_filetype('../../DataSets/electra_s7comm/output/attacked', 'csv')

attacked_dfs = []
normal_dfs = []
for file in  normal_files:
    df = pd.read_csv(file)
    normal_dfs.append(df)

for file in attacked_files:
    df = pd.read_csv(file)
    attacked_dfs.append(df)

combined_normal_df = pd.concat(normal_dfs)
combined_attacked_df = pd.concat(attacked_dfs)



print(combined_normal_df.head())
print(combined_attacked_df.head())

end = time.time()
print(f"total time: {end - start:.2f} secs")

# This is much slower than the multithreading approach
