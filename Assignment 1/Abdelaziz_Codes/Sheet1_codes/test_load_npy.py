import numpy as np

# Path to your .npy file
file_paths = [
    'QUT_attacked_taskE_block_0.npy',
    'QUT_attacked_taskE_block_1.npy',
    'QUT_attacked_taskE_block_2.npy',
              'QUT_attacked_taskE_block_28.npy',]
# Load the numpy file
datas = [np.load(file_path).flatten() for file_path in file_paths]
combined = np.concatenate(datas, axis=0)
print("Total number of loaded files:", len(datas), flush=True)
print("Shapes of loaded data arrays:", [d.shape for d in datas], flush=True)
print("Shape of combined data array:", combined.shape, flush=True)
# combine all loaded data into a single array if needed