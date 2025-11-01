'''
    @Author: Abdelaziz Neamatallah
    @Date: 23/10/2025
    @Desc: This is a utility file that contains helper functions to be used in the main tasks. 

'''

from scapy.all import rdpcap
import os, time
import pandas as pd
from pathlib import Path
import ipaddress
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import numpy as np
from itertools import combinations


output_dir = Path('../../DataSets/electra_s7comm/output')

#################### Multi-threading splitting of Electra dataset into normal and attacked files.
def create_pair_ip(row):
    try:
        ip1 = ipaddress.ip_address(row["sip"])
        ip2 = ipaddress.ip_address(row["dip"])
        return f"{min(ip1, ip2)}_{max(ip1, ip2)}"
    except:
        return None

def create_pair_mac(row):
    try:
        smac = row["smac"].lower()
        dmac = row["dmac"].lower()
        return f"{min(smac, dmac)}_{max(smac, dmac)}"
    except Exception:
        return None

def compute_packet_size(row):
    return 6*2 + 4*2 + 2*1 + row['data'] 


def process_chunk(chunk_id, chunk):
    chunk.Time = chunk.Time / 1_000_000
    chunk["packet_size"] = chunk['data']
    chunk["pair_ip"] = chunk.apply(create_pair_ip, axis=1)
    chunk["pair_mac"] = chunk.apply(create_pair_mac, axis=1)
    chunk_final = chunk[["Time","sip","dip","pair_ip","smac", "dmac", "pair_mac", "packet_size","label","request"]].sort_values("Time")

    normal = chunk_final[chunk_final['label']=="NORMAL"]
    attacked = chunk_final[chunk_final['label']!="NORMAL"]

    normal.to_csv(output_dir / f'normal/normal_data_{chunk_id}.csv', index=False, header=False)
    attacked.to_csv(output_dir / f'attacked/attacked_data_{chunk_id}.csv', index=False, header=False)
    return True


def multi_threading_splitting_electra(input_file='../../DataSets/electra_s7comm/electra_s7comm.csv'):
    start = time.time()
    chunks = pd.read_csv(input_file, chunksize=1_000_000)
    print("processing started")
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}

    end = time.time()
    print(f"total time: {end - start}")
#####################


##################### Multi-threading loading of QUT dataset and Electra dataset
def load_csv(file):
    return pd.read_csv(file)

def multithreading_loading_QUT(df_path): 

    start = time.time()

    # list files
    files = list_files_by_filetype(df_path, 'csv')
        
    # parallel read
    with ThreadPoolExecutor() as executor:
        normal_futures = {executor.submit(load_csv, f): f for f in files}

        normal_dfs = [f.result() for f in as_completed(normal_futures)]

    # combine
    combined_df = pd.concat(normal_dfs, ignore_index=True)

    print(f"Loaded {len(files)} files.")
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
    return combined_df





def load_csv_electra(file):
    # Use fast C parser and low_memory=False for better chunk merging
    return pd.read_csv(file, engine='c', low_memory=False)

def parallel_load(file_list, max_workers=None):
    dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_csv_electra, f): f for f in file_list}
        for f in as_completed(futures):
            try:
                dfs.append(f.result())
                print(f'append succedded from {futures[f]}')
            except Exception as e:
                print(f"Error loading {futures[f]}: {e}")
    return dfs

def multithreading_loading_electra(normal_file_dir='../../DataSets/electra_s7comm/output/normal', attacked_file_dir='../../DataSets/electra_s7comm/output/attacked'):
    start = time.time()
    print('started processing')
    normal_files = sorted(list_files_by_filetype(normal_file_dir, 'csv'))
    attacked_files = sorted(list_files_by_filetype(attacked_file_dir, 'csv'))
    first_normal_df = load_csv(normal_files[0])
    print(first_normal_df.head())
    first_attacked_df = load_csv(attacked_files[0])
    print(first_attacked_df.head())

    print('starting parallelization')
    # Use half available cores to avoid memory contention
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

######################

def manual_histogram(data, bins):
    min_val, max_val = min(data), max(data)
    print(min_val, max_val)
    bin_width = (max_val - min_val) / bins
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
    counts = [0] * bins # create a list for counts and initialze it with zeros. 

    for value in data:
        # Find the bin index
        idx = int((value - min_val) / bin_width)
        if idx == bins:  # Handle edge case where value == max_val
            idx -= 1
        counts[idx] += 1

    return bin_edges, counts


def generate_bytes_array_from_packet_list(pcap_files_path='../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set'):
    start = time.time()
    pcap_files = list_files_by_filetype(pcap_files_path, "pcap")
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
    return all_packets_array
def compute_chebyshev_distances_on_iat_optimized(flow):
    '''
        This is the most optimized way to compute Chebyshev distances, and it works because instead of iterating over each element and compare it with other elements until we find the maximum difference,
        we can simply find the global minimum and maximum of the flow, and then compute the distance of each element to these two extremes. The Chebyshev distance for each element is then the maximum of these two distances.
    '''
    iat = np.nan_to_num(flow["iat"].values) # ensure no NaN values
    global_min = np.min(iat)
    global_max = np.max(iat)
    # For 1D Chebyshev, each distance = max(|x - global_min|, |x - global_max|), and numpy uses the broadcasting feature to compute this for all elements in one go.
    return np.maximum(np.abs(iat - global_min), np.abs(iat - global_max))

def compute_chebyshev_distances_on_frame_len_optimized(flow):
    '''
        This is the most optimized way to compute Chebyshev distances, and it works because instead of iterating over each element and compare it with other elements until we find the maximum difference,
        we can simply find the global minimum and maximum of the flow, and then compute the distance of each element to these two extremes. The Chebyshev distance for each element is then the maximum of these two distances.
    '''
    frame_len = np.nan_to_num(flow["frame_len"].values) # ensure no NaN values
    global_min = np.min(frame_len)
    global_max = np.max(frame_len)
    # For 1D Chebyshev, each distance = max(|x - global_min|, |x - global_max|), and numpy uses the broadcasting feature to compute this for all elements in one go.
    return np.maximum(np.abs(frame_len - global_min), np.abs(frame_len - global_max))


def chebyshev_distance_for_raw_bytes(a, b):    
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)))
    b_padded = np.pad(b, (0, max_len - len(b)))
    return np.max(np.abs(a_padded.astype(np.int32) - b_padded.astype(np.int32)))


def compute_pair_distance(pair):
    a, b = pair
    return chebyshev_distance_for_raw_bytes(a, b)



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
    
    


# create traffic flows with time window 2 minutes, 4 minutes, and finally 6 minutes
def create_time_windowed_flows(flows, time_window_minutes):
    time_windowed_flows = {}
    print(time_window_minutes)
    time_delta = pd.Timedelta(minutes=time_window_minutes)
    
    for (app_proto, pair_id), group in flows.groupby(level=[0,1]):
        start_time = group['timestamp'].min()
        end_time = group['timestamp'].max()
        
        current_window_start = start_time
        while current_window_start < end_time:
            current_window_end = current_window_start + time_delta
            window_group = group[(group['timestamp'] >= current_window_start) & (group['timestamp'] < current_window_end)]
            
            if not window_group.empty:
                key = (app_proto, pair_id, current_window_start)
                time_windowed_flows[key] = window_group.reset_index(drop=True)
            
            current_window_start = current_window_end
            
    # convert the result to a DataFrame with MultiIndex
    df = pd.concat(time_windowed_flows.values(), keys=time_windowed_flows.keys())
    return df

def plot_histogram(bin_edges, counts, title, xlabel, ylabel, filename):
    plt.bar(
            [ (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts)) ],
            counts,
            width=(bin_edges[1] - bin_edges[0]),
            edgecolor='black'
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, dpi=300)


def process_single_pair(pair):
    pair_id, group, time_delta = pair
    time_windowed_flows = {}
    start_time = group['timestamp'].min()
    end_time = group['timestamp'].max()

    current_window_start = start_time
    while current_window_start < end_time:
        current_window_end = current_window_start + time_delta
        window_group = group[
            (group['timestamp'] >= current_window_start) &
            (group['timestamp'] < current_window_end)
        ]
        if not window_group.empty:
            key = (pair_id, current_window_start)
            time_windowed_flows[key] = window_group.reset_index(drop=True)
        current_window_start = current_window_end

    return time_windowed_flows

def create_time_windowed_flows_electra(flows, time_window_minutes):
    print(f" Creating {time_window_minutes}-minute windows...")
    time_delta = pd.Timedelta(minutes=time_window_minutes)
    time_windowed_flows = {}

    grouped = list(flows.groupby(level=[0]))  # [(pair_id, group_df), ...]

    # Use parallel processing
    max_workers = max(1, os.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pair, (pid, g, time_delta)): pid for pid, g in grouped}
        for f in as_completed(futures):
            try:
                result = f.result()
                time_windowed_flows.update(result)
            except Exception as e:
                print(f" Error processing pair {futures[f]}: {e}")

    # Merge all processed results into a single DataFrame
    df = pd.concat(time_windowed_flows.values(), keys=time_windowed_flows.keys())
    print(f" Completed {time_window_minutes}-minute flow creation using {max_workers} cores.")
    return df

