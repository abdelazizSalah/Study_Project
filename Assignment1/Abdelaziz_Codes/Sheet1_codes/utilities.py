'''
    @Author: Abdelaziz Neamatallah
    @Date: 23/10/2025
    @Desc: This is a utility file that contains helper functions to be used in the main tasks. 

'''
TESTING = True
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scapy.all import rdpcap
import os, time
import pandas as pd
from pathlib import Path
import ipaddress
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor,as_completed
import numpy as np
import matplotlib.pyplot as plt
output_dir = Path('../../DataSets/electra_s7comm/output')

#################### Multi-threading splitting of Electra dataset into normal and attacked files.
def create_pair_ip(row):
    """
    The function `create_pair_ip` takes a row of data containing source and destination IP addresses,
    creates a pair of IP addresses, and returns them in a specific format.
    
    :param row: The function takes a row of data as input. The row is expected to
    contain two IP addresses: "sip" (source IP) and "dip" (destination IP). The function then creates a
    pair of IP addresses in a specific format and returns it as a string.
    """
    try:
        ip1 = ipaddress.ip_address(row["sip"])
        ip2 = ipaddress.ip_address(row["dip"])
        return f"{min(ip1, ip2)}_{max(ip1, ip2)}"
    except:
        return None

def create_pair_mac(row):
    '''
    Same as create_pair_ip but for mac addresses
    '''
    try:
        smac = row["smac"].lower()
        dmac = row["dmac"].lower()
        return f"{min(smac, dmac)}_{max(smac, dmac)}"
    except Exception:
        return None

def compute_packet_size(row):
    """
    The function `compute_packet_size` calculates the total size of a packet based on the data provided
    in a row from electra Dataset. 
    
    :param row: The `row` parameter is a DataFrame row.
    :return: The function is returning the total size of a packet based on the
    data provided in the `row` parameter. 
    The calculation includes 6*2 bytes for a Mac addresses (smac and dmac) of data,
    4*2 bytes for Ip addresses (sip and dip), 2 bytes for two booleans, and the size of the `data`
    field in the `row` which is the payload.
    """
    return 6*2 + 4*2 + 2*1 + row['data'] 

def determine_direction(row, src_col_name, pair_index):
    '''
        My Logic is that, I will sort the flows with timestamp, and then the first packet will be from the source to destination which is sent,
        so I will assign it 0, and the other packets will be assigned based on this logic.
        if the src_ip is same as the first ip in the pair, then it is 0, else 1.
    '''
    src_ip = row[src_col_name]
    pair_id = row.name[pair_index]  # Accessing pair_id from the MultiIndex
    first_ip_in_pair = pair_id.split('__')[0]
    if src_ip == first_ip_in_pair:
        return 0
    else:
        return 1


def process_chunk(chunk_id, chunk):
    """
    The function `process_chunk` processes a chunk of data by manipulating and sorting columns,
    separating normal and attacked data, and saving the results to CSV files.
    
    :param chunk_id: The `chunk_id` parameter is used to identify the specific chunk of data being
    processed. It is a unique identifier or index for the chunk
    :param chunk: The `chunk` parameter is a DataFrame containing network packet data. The
    function processes this chunk of data by performing various operations like
    converting time units, creating new columns, sorting the data, filtering normal and attacked
    packets, and then saving the processed data into separate CSV files for
    """
    chunk.Time = chunk.Time / 1_000_000
    chunk["packet_size"] = chunk['data']
    chunk["pair_ip"] = chunk.apply(create_pair_ip, axis=1)
    chunk["pair_mac"] = chunk.apply(create_pair_mac, axis=1)
    chunk_final = chunk[["Time","sip","dip","pair_ip","smac", "dmac", "pair_mac", "packet_size","label","request"]].sort_values("Time")

    normal = chunk_final[chunk_final['label']=="NORMAL"]
    attacked = chunk_final[chunk_final['label']!="NORMAL"]

    normal.to_csv(output_dir / f'normal/normal_data_{chunk_id}.csv', index=False, header=False)
    attacked.to_csv(output_dir / f'attacked/attacked_data_{chunk_id}.csv', index=False, header=False)


def multi_threading_splitting_electra(input_file='../../DataSets/electra_s7comm/electra_s7comm.csv'):
    """
    The function `multi_threading_splitting_electra` reads a CSV file in chunks, processes each chunk
    using multiple processes, and measures the total processing time.
    
    :param input_file: The `input_file` is the file path to the CSV file.
      
    """
    start = time.time()
    chunks = pd.read_csv(input_file, chunksize=1_000_000)
    print("processing started")
    '''
        ProcessPoolExecutor is a method used for parallel processing, it creates many processes based on the maxWorker parameter, and assign each process
        a chunk to process it, this is useful when we have large data that needs to be processed quickly.
    '''
    maxWorkers = max(1, os.cpu_count())
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        futures = {executor.submit(process_chunk, i, c): i for i, c in enumerate(chunks)}

    end = time.time()
    print(f"total time: {end - start}")
#####################


##################### Multi-threading loading of QUT dataset and Electra dataset
def load_csv(file):
    if TESTING:
        return pd.read_csv(file, nrows = 1000 )
    else:
        return pd.read_csv(file)

def multithreading_loading_QUT(df_path): 
    """
    The function `multithreading_loading_QUT` loads multiple CSV files concurrently using multithreading
    and combines them into a single DataFrame.
    
    :param df_path: The `df_path` parameter in the `multithreading_loading_QUT` function is the path to
    the directory where the CSV files are located. The function reads all CSV files from this directory
    in parallel using multithreading, combines them into a single DataFrame, and returns the combined
    DataFrame
    :return: The function `multithreading_loading_QUT` returns a combined DataFrame that is created by
    concatenating DataFrames loaded from multiple CSV files in parallel using multithreading. The
    function also prints the number of files loaded and the total time taken for the operation.
    """

    start = time.time()

    # list files
    files = list_files_by_filetype(df_path, 'csv')
        
    # parallel read
    maxWorkers = max(1, os.cpu_count())
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
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
    if TESTING:
        return pd.read_csv(file, engine='c', low_memory=False, nrows=5000)
    else:
        return pd.read_csv(file, engine='c', low_memory=False)

def parallel_load(file_list, maxWorkers=None):
    """
    The function `parallel_load` uses a ProcessPoolExecutor to load multiple CSV files concurrently and
    returns a list of DataFrames.
    
    :param file_list: The `file_list` parameter is a list of file paths that we want to load
    concurrently using parallel processing. 
    :param maxWorkers: The `maxWorkers` parameter in the `parallel_load` function specifies the
    maximum number of worker processes to use for parallel execution. If `maxWorkers` is set to `None`,
    then the `ProcessPoolExecutor` will use the default number of worker processes, which is typically
    the number of CPU
    :return: The function `parallel_load` returns a list of DataFrames that are loaded from CSV files
    using the `load_csv_electra` function in parallel.
    """
    dfs = []
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
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
    # I sort the files to be able to read them in order especially the first file which contains the header.
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
    print(' All DataFrames appended successfully.')


    print(f"Loaded {len(normal_files)} normal and {len(attacked_files)} attacked files.")
    print(f"Total time: {time.time() - start:.2f} seconds")

######################

def manual_histogram(data, bins):
    """
    The function `manual_histogram` calculates the histogram of a given dataset by dividing it into
    specified number of bins.
    
    :param data: Data is the list of numerical values for which we want to create a histogram
    :param bins: The `bins` parameter in the `manual_histogram` function represents the number of
    intervals (or bins) that you want to divide your data into when creating a histogram. Increasing the
    number of bins will result in a more detailed histogram with smaller intervals, while decreasing the
    number of bins will result in a
    :return: The function `manual_histogram` returns two lists: `bin_edges` which contains the edges of
    the bins, and `counts` which contains the frequency counts of the data points within each bin.
    """
    min_val, max_val = min(data), max(data)
    print(min_val, max_val)
    bin_width = (max_val - min_val) / bins
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)] # 0, bin_width, 2*bin_width, ..., bins*bin_width
    counts = [0] * bins # create a list for counts and initialze it with zeros. 

    for value in data:
        # Find the bin index
        idx = int((value - min_val) / bin_width)
        if idx == bins:  # Handle edge case where value == max_val
            idx -= 1
        counts[idx] += 1

    return bin_edges, counts



def read_pcap_as_byte_sequences(pcap_path):
    print(f' the given path is {pcap_path}')
    # check if the file is empty
    if os.path.getsize(pcap_path) == 0:
        print(f'file is found empty at path {pcap_path}')
        return []
    

    packets = rdpcap(pcap_path)              # Load all packets in form of a list.
    return [bytes(pkt) for pkt in packets]  # Convert each packet to raw bytes
    
    


def generate_bytes_array_from_packet_list(pcap_files_path='../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set', label = 'control', pad = True):
    """
    This function reads packet data from pcap files, processes them using multithreading, converts them
    into a NumPy array, and saves the array to a file.
    
    :param pcap_files_path: The `pcap_files_path` parameter is the path to the directory containing the
    pcap files from which you want to generate a bytes array. In this function, the default path is set
    to `'../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813, defaults to
    ../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set (optional)
    :return: The function `generate_bytes_array_from_packet_list` returns the NumPy array
    `all_packets_array` containing the byte sequences read from the pcap files processed in the
    function.
    """
    start = time.time()
    pcap_files = list_files_by_filetype(pcap_files_path, "pcap")
    print(f'number of pcap files to be processed: {pcap_files}')
    if TESTING: 
        pcap_files = [pcap_files[0]] if label == 'control' else pcap_files  # for testing, use only first pcap file

    print(f'the pcap file to be processed {pcap_files}')
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

    # ensure that all packets all of the same length
    max_len = max(len(pkt) for pkt_list in all_packets for pkt in pkt_list)

    print('padding the packets')
    # padd all of them to the max length
    if pad:
        for i in range(len(all_packets)):
            all_packets[i] = [pkt.ljust(max_len, b'\0') for pkt in all_packets[i]]
        
    min_len = min(len(pkt) for pkt_list in all_packets for pkt in pkt_list)
    print(f'max packet len: {max_len}, min packet len: {min_len}')


    print(f'time taken to process all files {time.time() - start}')
    start = time.time()
    # Convert to large NumPy array with dtype=object (packets may differ in length)
    all_packets_array = np.array(all_packets, dtype=object)
    np.save(f'all_packets_{label}.npy', all_packets_array)

    print(f'time taken to conver and write all files {time.time() - start}')
    start = time.time()

    if TESTING:
        all_packets_array = all_packets_array[:10]  # for testing, use only first 2 pcap files
        print(type(all_packets_array[0]))
        print(type(all_packets_array))

    return all_packets_array


def unoptimized_compute_chebyshev_distances(flow):
    '''
    I have traffic flow 1 in this format: "timestamp, src_ip, dst_ip, frame_len, iat"
    Then we have all other traffic flows in the same format in the same window interval. 
    so to compute chybyshev distance between flow 1 and all other flows, we need to extract the iat column for each flow and then compute the distance.
    the distance is as follows: 
      # chybyshev_t1 = max(|t1_iat - ti_iat|) for i in range(n)
    I should have 3574782 chybyshev distances computed
    
    '''
    chebyshev_distances = []
    for i in range(len(flow)): 
        current_flow_iat = flow['iat'][i]
        max_distance = -1000000 
        for j in range( i + 1, len(flow)):
            other_flow_iat = flow['iat'][j]
            max_distance = max(abs(current_flow_iat - other_flow_iat), max_distance)
        chebyshev_distances.append(max_distance)
    return chebyshev_distances # this takes too long, so I need to optimize it



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
    """
    The function calculates the Chebyshev distance between two raw byte arrays by padding them to equal
    length and finding the maximum absolute difference between corresponding elements.
    
    :param a: The `a` parameter in the `chebyshev_distance_for_raw_bytes` function represents one of the
    raw byte arrays for which you want to calculate the Chebyshev distance
    :param b: It seems like you have provided the function definition for calculating the Chebyshev
    distance between two raw byte arrays `a` and `b`. However, you have not provided the value for the
    variable `b`. Could you please provide the value of `b` so that I can assist you further
    :return: The function `chebyshev_distance_for_raw_bytes` calculates the Chebyshev distance between
    two raw byte arrays `a` and `b`. It pads the arrays to equal length, converts them to `int32` data
    type, calculates the absolute difference element-wise, and then returns the maximum absolute
    difference between corresponding elements in the two arrays.
    """
    max_len = max(len(a), len(b))
    a_padded = np.pad(a, (0, max_len - len(a)))
    b_padded = np.pad(b, (0, max_len - len(b)))
    return np.max(np.abs(a_padded.astype(np.int32) - b_padded.astype(np.int32)))


def compute_pair_distance(pair):
    a, b = pair
    return chebyshev_distance_for_raw_bytes(a, b)



def normalize_packets(packets):
    """
    The function `normalize_packets` takes a list of packets which may contain different values and representation of the data and converts them into numpy
    arrays of unsigned integers to make all the data consistent and to be able to compute chybeshev distance.
    
    :param packets: The `packets` parameter is a list containing packets of data. Each packet can be of
    different types such as bytes, bytearray, list of byte strings, numpy array, or any other data type
    that can be converted to a numpy array of unsigned integers (uint8). The function `normalize_packets
    :return: The `normalize_packets` function returns a list of NumPy arrays where each element in the
    list is a normalized version of the input packets. The normalization process involves converting the
    packets into NumPy arrays of unsigned 8-bit integers (uint8).
    """
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


# create traffic flows with time window 2 minutes, 4 minutes, and finally 6 minutes
def create_time_windowed_flows(flows, time_window_minutes):
    """
    The function `create_time_windowed_flows` takes a DataFrame of flows with timestamps and groups them
    into time windows of a specified duration, returning a new DataFrame with a MultiIndex.
    
    :param flows: The `flows` parameter in the `create_time_windowed_flows` function is likely a
    DataFrame containing flow data with columns like 'app_proto', 'pair_id', and 'timestamp'. The
    function processes this data to create time-windowed flows based on the specified time window in
    minutes
    :param time_window_minutes: The `time_window_minutes` parameter in the `create_time_windowed_flows`
    function represents the size of the time window in minutes that you want to use for grouping the
    flows. This parameter determines the duration of each time window within which the flows will be
    aggregated and processed
    :return: The function `create_time_windowed_flows` returns a DataFrame containing time-windowed
    flows data. The data is grouped based on the specified time window in minutes and includes
    information such as application protocol, pair ID, and timestamps within each time window.
    """
    time_windowed_flows = {}
    print(time_window_minutes)
#    time_delta = pd.Timedelta(minutes=time_window_minutes)
 #   print(time_delta)    
    time_delta = time_window_minutes * 60
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
            [ (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts)) ], # I set the bar in the middle between the two edges this is the X axis
            counts,
            width=(bin_edges[1] - bin_edges[0]), # width = the bin size. 
            edgecolor='black'
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, dpi=300)


def process_single_pair(pair):
    """
    The function `process_single_pair` takes a pair containing an ID, a group of data, and a time delta,
    and creates time-windowed flows based on the timestamp within the group data.
    
    :param pair: The `process_single_pair` function takes a tuple `pair` as input, which contains three
    elements: `pair_id`, `group`, and `time_delta`. The function processes the data in the `group`
    DataFrame by splitting it into time windows based on the `time_delta` value
    :return: The function `process_single_pair` returns a dictionary `time_windowed_flows` where the
    keys are tuples of `pair_id` and `current_window_start`, and the values are DataFrames containing
    the data from the input `group` DataFrame that fall within each time window defined by `time_delta`.
    """
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
    """
    The function `create_time_windowed_flows_electra` processes flows data in time windows using
    parallel processing and returns a DataFrame.
    
    :param flows: The `flows` parameter in the `create_time_windowed_flows_electra` function is likely a
    DataFrame containing flow data. This data could represent information about flows between pairs,
    possibly in a network or system. The function seems to be designed to create time-windowed flows
    based on the input
    :param time_window_minutes: The `time_window_minutes` parameter in the
    `create_time_windowed_flows_electra` function represents the size of the time windows in minutes
    that you want to create for the flows data. This parameter determines the duration of each time
    window for grouping the flows data
    :return: The function `create_time_windowed_flows_electra` returns a DataFrame containing the
    time-windowed flows data after processing the input flows data using parallel processing with a
    specified time window in minutes.
    """
    print(f" Creating {time_window_minutes}-minute windows...")
    #time_delta = pd.Timedelta(minutes=time_window_minutes)
    time_delta = time_window_minutes * 60
    time_windowed_flows = {}

    grouped = list(flows.groupby(level=[0]))  # [(pair_id, group_df), ...]
    print(f"max timestamp: {max(flows['timestamp'])}")
    # Use parallel processing
    max_workers = max(1, os.cpu_count())
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


def compute_best_tsne(flow, perplexities, learning_rates, index, label):
    X = np.nan_to_num(StandardScaler().fit_transform(flow[['frame_len', 'iat']].values))
    best_score, best_tsne = float('inf'), None
    for p in perplexities:
        for lr in learning_rates:
            tsne = TSNE(n_components=2, perplexity=p, learning_rate=lr, random_state=42)
            embedding = tsne.fit_transform(X)
            if tsne.kl_divergence_ < best_score:
                best_score, best_tsne = tsne.kl_divergence_, embedding
    color_map = {'master': 'green', 'hmi': 'blue'}
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color)
                for label, color in color_map.items()]
    if label != 'control':
        attack_labels = flow['attack_label']
        colors = attack_labels.map(color_map).fillna('red')
        # add to handles red for attacks 
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label='other attacks', markerfacecolor='red'))

    plt.legend(handles=handles, title="Attack Type")
    plt.figure(figsize=(6,5))
    plt.scatter(best_tsne[:,0], best_tsne[:,1], s=10, alpha=0.7, c=colors if label != 'control' else 'gray')
    plt.title(f"t-SNE (control) - {(index+1)*2} min\nBest KL: {best_score:.4f}")
    plt.savefig(f"tsne_control_{(index+1)*2}min.png", dpi=300)
    plt.close()
    return best_score



# ---- single-GPU worker ----
