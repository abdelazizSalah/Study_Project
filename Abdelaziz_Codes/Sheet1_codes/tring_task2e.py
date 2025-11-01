from scapy.all import rdpcap
import os, time
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import numpy as np
from itertools import combinations

#list all files of certain filetype from directory and it's subdirectories
def list_files_by_filetype(root_path, filetype):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith("."+filetype):
                full_path = os.path.join(dirpath, filename)
                pcap_files.append(full_path)
    return pcap_files

def read_pcap_as_byte_sequences(pcap_path):
    packets = rdpcap(pcap_path)              # Load all packets
    return [bytes(pkt) for pkt in packets]  # Convert each packet to raw bytes
    
    

def chebyshev_distance(a, b):
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)))
    b_padded = np.pad(b, (0, max_len - len(b)))
    return np.max(np.abs(a_padded.astype(np.int32) - b_padded.astype(np.int32)))

def compute_pair_distance(pair):
    a, b = pair
    return chebyshev_distance(a, b)



def normalize_packets(packets):
    normalized = []
    for p in packets:
        if isinstance(p, (bytes, bytearray)):
            arr = np.frombuffer(p, dtype=np.uint8)
        elif isinstance(p, list) and isinstance(p[0], (bytes, bytearray)):
            # convert list of byte strings → flat uint8 array
            arr = np.frombuffer(b"".join(p), dtype=np.uint8)
        elif isinstance(p, np.ndarray):
            arr = p.astype(np.uint8)
        else:
            arr = np.array(p, dtype=np.uint8)
        normalized.append(arr)
    return normalized

# Example usage
if __name__ == "__main__":
    # pcap_file = "E:/GitHub/Study_Project/DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7Comm/control/hmi.pcap.csv"
    '''
    start = time.time()
    pcap_files = list_files_by_filetype('../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set', "pcap")
    print(pcap_files)
    all_packets = []
    print(f'time taken to load all list all files {time.time() - start}')
    start = time.time()
    maxWorkers = os.cpu_count() or 4
    print(f'max workers: {maxWorkers}\n now creating packet list...')
    with ThreadPoolExecutor(max_workers= maxWorkers) as executor:
        for packet_lists in executor.map(read_pcap_as_byte_sequences, pcap_files):
            all_packets.append(packet_lists)
            print(f'packet len: {len(packet_lists)}')
            print(packet_lists[:5], sep='\n')


    print(f'time taken to process all files {time.time() - start}')
    start = time.time()
    # Convert to large NumPy array with dtype=object (packets may differ in length)
    all_packets_array = np.array(all_packets, dtype=object)
    np.save('all_packets.npy', all_packets_array)

    print(f'time taken to conver and write all files {time.time() - start}\n now creating pairs')
    start = time.time()
    '''
     #Later to read the created file 
	     
    # Later load them instantly
    start = time.time()
    all_packets_array = np.load("all_packets.npy", allow_pickle=True)
    print(f"Loaded {len(all_packets_array)} arrays successfully.")
    print(f'time taken to load all npfile {time.time() - start}\n now creating pairs')

    all_packets =  [] 
    for array in all_packets_array: 
        print(len(array))
        all_packets.extend(arr for arr in array)
    print(type(all_packets[0]))
    print(len(all_packets))
    #print(f"Total packets combined: {len(all_packets)}")
    all_packets_array = normalize_packets(all_packets)

    
    # Generate all packet pairs
    start = time.time()
    pairs = list(combinations(all_packets_array[:100], 2))

    print(f'time taken to generate all pairs {time.time() - start} with {len(pairs)}, \n now computing distances')
    start = time.time()
    # Parallel computation
    maxWorkers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        distances = list(executor.map(compute_pair_distance, pairs))

    print(f'time taken to compute chebychev distance files {time.time() - start}')
    start = time.time()
    print(f"Computed {len(distances)} Chebyshev distances in parallel.\n now saving to file the distances")


    # Save to file (binary .npy for fast I/O)
    output_path = "chebyshev_distances.npy"
    np.save(output_path, np.array(distances, dtype=np.float32))
    print(f"Saved distances to {output_path}")

    print(f'time taken to write chebyshev distance to file {time.time() - start}')
    #start = time.time()
    print(len(distances))
    print(distances[:100])
