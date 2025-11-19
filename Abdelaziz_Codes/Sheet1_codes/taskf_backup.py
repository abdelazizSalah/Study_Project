import numpy as np
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

def worker(file_path):
    try:
        data = np.load(file_path)
        local_hist, _ = np.histogram(data, bins=5, range=(0, 255))
        print(f"[✓] Finished {file_path}")
        return local_hist
    except Exception as e:
        print(f"[✗] Skipping {file_path}: {e}")
        return np.zeros(5, dtype=int)

if __name__ == "__main__":
    folder = "chebyshev_blocks"
    files = [os.path.join(folder, f) for f in os.listdir(folder) if "QUT_attacked_taskE" in f]
    # Use pool to limit open files
    with mp.Pool(processes=os.cpu_count()) as pool:
        results = pool.map(worker, files)

    global_hist = np.sum(results, axis=0)

    bin_edges = np.linspace(0, 255, 6)
    bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i in range(5)]

    plt.figure(figsize=(8,4))
    plt.bar(np.arange(5), global_hist, color='steelblue')
    plt.xticks(np.arange(5), bin_labels, rotation=30)
    plt.title('Global Histogram (5 bins) of Chebyshev Distances – QUT Attacked Task E')
    plt.xlabel('Distance Range')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('global_histogram_QUT_attacked_taskE_5bins.png')
    plt.close()
    print(f'number of files {len(files)}')
    print("[✓] Histogram saved as 'global_histogram_QUT_attacked_taskE_5bins.png'")

