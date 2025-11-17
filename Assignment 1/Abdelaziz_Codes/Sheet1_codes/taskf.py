# import numpy as np
# import os
# import multiprocessing as mp
# import matplotlib.pyplot as plt

# def worker(file_path, shared_hist, lock):
#     print(f"[+] Processing {file_path}")
#     data = np.load(file_path)
#     local_hist, _ = np.histogram(data, bins=256, range=(0, 255))
#     with lock:  # prevent race conditions
#         for i in range(256):
#             shared_hist[i] += int(local_hist[i])
#     print(f"[✓] Finished {file_path}")

# if __name__ == "__main__":
#     folder = "chebyshev_blocks"
#     files = [os.path.join(folder, f) for f in os.listdir(folder) if "QUT_attacked_taskE" in f]

#     # ✅ Use mp.Array (shared memory) + Lock
#     shared_hist = mp.Array('q', 256)  # 'q' = 64-bit int, safer for large counts
#     lock = mp.Lock()

#     # ⚠ On Windows, Pool can't send shared memory safely → use manual Process
#     processes = []
#     for f in files:
#         p = mp.Process(target=worker, args=(f, shared_hist, lock))
#         p.start()
#         processes.append(p)

#     for p in processes:
#         p.join()

#     # ✅ Convert shared memory to numpy array
#     global_hist = np.frombuffer(shared_hist.get_obj(), dtype=np.int64)

    
#     # ✅ Plot 5 bins only
#     bin_edges = np.linspace(0, 255, 6)[:-1]
#     plt.figure(figsize=(8,4))
#     plt.bar(bin_edges, global_hist, width=50, color='steelblue', align='edge')
#     plt.title('Global Histogram (5 bins) of Chebyshev Distances – QUT Attacked Task E')
#     plt.xlabel('Distance Range')
#     plt.ylabel('Frequency')
#     plt.tight_layout()
#     plt.savefig('global_histogram_QUT_attacked_taskE_5bins.png')
#     plt.close()


#     print("[✓] Histogram saved as 'global_histogram_QUT_attacked_taskE.png'")

import numpy as np
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

def worker(file_path, shared_hist, lock):
    print(f"[+] Processing {file_path}")
    data = np.load(file_path)
    local_hist, _ = np.histogram(data, bins=5, range=(0, 255))
    with lock:
        for i in range(5):
            shared_hist[i] += int(local_hist[i])
    print(f"[✓] Finished {file_path}")

if __name__ == "__main__":
    folder = "chebyshev_blocks"
    files = [os.path.join(folder, f) for f in os.listdir(folder) if "QUT_attacked_taskE" in f]

    shared_hist = mp.Array('q', [0]*5)
    lock = mp.Lock()

    processes = []
    for f in files:
        p = mp.Process(target=worker, args=(f, shared_hist, lock))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    global_hist = np.frombuffer(shared_hist.get_obj(), dtype=np.int64, count=5)

    # ✅ Generate correct bin edges and labels
    bin_edges = np.linspace(0, 255, 6)  # 6 edges → 5 bins
    bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i in range(5)]

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(5), global_hist, color='steelblue')
    plt.xticks(np.arange(5), bin_labels, rotation=30)
    plt.title('Global Histogram (5 bins) of Chebyshev Distances – QUT Attacked Task E')
    plt.xlabel('Distance Range')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('global_histogram_QUT_attacked_taskE_5bins.png')
    plt.close()

    print("[✓] Histogram saved as 'global_histogram_QUT_attacked_taskE_5bins.png'")
