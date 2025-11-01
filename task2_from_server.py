# adding Anna's utility functions
import sys, time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed


import numpy as np
sys.path.append('../../Anna_Code')  
from file_helper import *

# I noticed that  the combined file I generated is twice the size of the normal electra file, so most probably I should work on the normal electra file directly :)

# Task2 Creating traffic flow for attacker and normal traffic from QUT_S7Comm dataset pcaps
    
def load_csv(file):
    return pd.read_csv(file)

def multithreading_loading(df_path): 

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



def task2a_preprocessing_QUT(df_path, label):
    # loading the attack and normal datasets from csv files
    loaded_df = multithreading_loading(df_path)
    # selecting the important features
    df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id']]

    # drop any packet with undefined app_proto and any packet with no src_ip or dst_ip
    df_selected_features = df_selected_features[df_selected_features['app_proto'] != 'undetected:-1:-1']
    df_selected_features = df_selected_features[df_selected_features['src_ip'].notna() & df_selected_features['dst_ip'].notna()]
    # group packets by application protocol
    df_grouped = df_selected_features.groupby(['app_proto', 'pair_id'])

    # selecting interesting features and sorting them by timestamp
    print('before sorting')
    flows = df_grouped.apply(lambda x: x[['timestamp', 'src_ip', 'dst_ip', 'frame_len']].sort_values('timestamp').reset_index(drop=True)) # Reseting the index to get new indicies after sorting to avoid confusion.
    print('after sorting')
    # converting timestamp to datetime
#    flows['timestamp'] = pd.to_datetime(flows['timestamp'], unit='s')
    
    # adding inter-arrival time between packets in the same flow
    flows['iat'] = flows.groupby(level=[0,1])['timestamp'].diff().dt.total_seconds() # levels here refers to the index of the groups

    # creating time_widnowed_flows with 2, 4, and 6 mins
    flows_2min = create_time_windowed_flows(flows, 2)
    flows_4min = create_time_windowed_flows(flows, 4)
    flows_6min = create_time_windowed_flows(flows, 6)
    # printing thier size
    print(f" data from {label} \n flows_2min len: {len(flows_2min)},  flows_4min len: {len(flows_4min)},  flows_6min len: {len(flows_6min)}.")

    # printing number of flows per protocol
    print(f"{label} number of flows per protocol\n 2min: {flows_2min.index.get_level_values(0).value_counts()}\n 4min: {flows_4min.index.get_level_values(0).value_counts()}\n 6min: {flows_6min.index.get_level_values(0).value_counts()}") # this is the number of flows per application protocol, and if we summed them up, it should be equal to the total number of flows we got earlier.

    return flows_2min, flows_4min, flows_6min

def task2a_preprocessing_Electra(df_path, label):
    # Time,smac,dmac,sip,dip,request,fc,error,address,data,label -> headers
    # loading the attack and normal datasets from csv files
    print('task2a starting')
    start = time.time()
    loaded_df = load_csv(df_path)
    end = time.time()
    print(f'loading dataset in {end - start} secs')
    print(loaded_df.head())    
    # change column name Time to timestamp
    loaded_df.rename(columns={'Time': 'timestamp', 'packet_size':'frame_len'}, inplace=True)
    print(loaded_df.columns)

    # selecting the important features
    df_selected_features = loaded_df[['timestamp', 'sip', 'dip', 'frame_len', 'pair_ip']]
    print('after selecting the features')
    # drop any packet with undefined app_proto and any packet with no src_ip or dst_ip
    # df_selected_features = df_selected_features[df_selected_features['sip'].notna() & df_selected_features['dip'].notna()]
    # group packets by application protocol
    df_grouped = df_selected_features.groupby('pair_ip')
    print(df_grouped.head())
    print('before grouping')
    # selecting interesting features and sorting them by timestampstamp
    flows = df_grouped.apply(lambda x: x[['timestamp', 'sip', 'dip', 'frame_len']].sort_values('timestamp').reset_index(drop=True)) # Reseting the index to get new indicies after sorting to avoid confusion.

    # converting timestamp to datetimestamp
    flows['timestamp'] = pd.to_datetime(flows['timestamp'], unit='s') # not sure if this will be a good idea.
    print('after timestamp conversion')
    # adding inter-arrival timestamp between packets in the same flow
    flows['iat'] = flows.groupby(level=[0])['timestamp'].diff().dt.total_seconds() # levels here refers to the index of the groups
    print('adding iat')
    print('starting creating flows 2 mins')

    # writing the file to save time
    print(f'writing csv file {label}0.csv')
    now = time.time()
    flows.to_parquet(label + '.parquet', index=False)   
#    flows.to_csv(label)
    print(f'took too long: {time.time() - now}')
    # creating time_widnowed_flows with 2, 4, and 6 mins
    flows_2min = create_time_windowed_flows_electra(flows, 2)
    
    print('starting creating flows 4 mins')
    flows_4min = create_time_windowed_flows_electra(flows, 4)

    print('starting creating flows 6 mins')
    flows_6min = create_time_windowed_flows_electra(flows, 6)
    # printing thier size
    print(f" data from {label} \n flows_2min len: {len(flows_2min)},  flows_4min len: {len(flows_4min)},  flows_6min len: {len(flows_6min)}.")

      # printing number of flows per protocol
    print(f"{label} number of flows per protocol\n 2min: {flows_2min.index.get_level_values(0).value_counts()}\n 4min: {flows_4min.index.get_level_values(0).value_counts()}\n 6min: {flows_6min.index.get_level_values(0).value_counts()}") # this is the number of flows per application protocol, and if we summed them up, it should be equal to the total number of flows we got earlier.

    return flows_2min, flows_4min, flows_6min

def task2b (flows_2min, flows_4min, flows_6min): 
    print('start processing task2b')
    # computing chebyshev_distance using iat
    chebyshev_distances_2min = compute_chebyshev_distances_memory_safe_on_iat(flows_2min)
    chebyshev_distances_4min = compute_chebyshev_distances_memory_safe_on_iat(flows_4min)
    chebyshev_distances_6min = compute_chebyshev_distances_memory_safe_on_iat(flows_6min)

    # print on frame_len column
    chebyshev_distances_2min_frame_len = compute_chebyshev_distances_memory_safe_on_frame_len(flows_2min)
    chebyshev_distances_4min_frame_len = compute_chebyshev_distances_memory_safe_on_frame_len(flows_4min)
    chebyshev_distances_6min_frame_len = compute_chebyshev_distances_memory_safe_on_frame_len(flows_6min)

    return chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len

def task2c(chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len, label):
    print('start processing task2c')
    iter = 2
    for data in [chebyshev_distances_2min, chebyshev_distances_4min, chebyshev_distances_6min]:
        bin_edges, counts = manual_histogram(data, bins=4)
        plt.bar(
            [ (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts)) ],
            counts,
            width=(bin_edges[1] - bin_edges[0]),
            edgecolor='black'
        )
        plt.title(f"Chebyshev Distances {label} Traffic")
        plt.xlabel("Chebyshev Distance (IAT)")
        plt.ylabel("Frequency")
        plt.savefig(f"iat_{iter}_{label}.jpg", dpi=300)
        iter+=2
 #       plt.show()
    iter = 0
    for data in [chebyshev_distances_2min_frame_len, chebyshev_distances_4min_frame_len, chebyshev_distances_6min_frame_len]:
        bin_edges, counts = manual_histogram(data, bins=4)
        plt.bar(
            [ (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts)) ],
            counts,
            width=(bin_edges[1] - bin_edges[0]),
            edgecolor='black'
        )
        plt.title(f"Chebyshev Distances {label} Traffic")
        plt.xlabel("Chebyshev Distance (Frame Length)")
        plt.ylabel("Frequency")
        plt.savefig(f"frameLen_{iter}_{label}.jpg", dpi=300)
        iter+= 2
#        plt.show()


def processing_QUT_loaded_dataframe(df_path ='../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/', label='normal'): 
    ### Task 2.a: Creating the flows
    flows_2min, flows_4min, flows_6min = task2a_preprocessing_QUT(df_path, label)
    ### Task2.b: Computing Chebyshev Distance between flows. 
    chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len = task2b(flows_2min, flows_4min, flows_6min )
    ### Task2.c Visualizing Histograms    
    task2c (chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len, label)


def processing_Electra_loaded_dataframe(df_path ='../../DataSets/electra_s7comm/output/attacked', label='normal'):
    ### Task 2.a: Creating the flows
    flows_2min, flows_4min, flows_6min = task2a_preprocessing_Electra(df_path, label)
    ### Task2.b: Computing Chebyshev Distance between flows. 
    chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len = task2b(flows_2min, flows_4min, flows_6min )
    ### Task2.c Visualizing Histograms    
    task2c (chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len, label)


def manual_histogram(data, bins):
    min_val, max_val = min(data), max(data)
    print(min_val, max_val)
    bin_width = (max_val - min_val) / bins
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
    counts = [0] * bins

    for value in data:
        # Find the bin index
        idx = int((value - min_val) / bin_width)
        if idx == bins:  # Handle edge case where value == max_val
            idx -= 1
        counts[idx] += 1

    return bin_edges, counts


def compute_chebyshev_distances_memory_safe_on_iat(flow):
    '''
        This is the most optimized way to compute Chebyshev distances, and it works because instead of iterating over each element and compare it with other elements until we find the maximum difference,
        we can simply find the global minimum and maximum of the flow, and then compute the distance of each element to these two extremes. The Chebyshev distance for each element is then the maximum of these two distances.
    '''
    iat = np.nan_to_num(flow["iat"].values) # ensure no NaN values
    global_min = np.min(iat)
    global_max = np.max(iat)
    # For 1D Chebyshev, each distance = max(|x - global_min|, |x - global_max|), and numpy uses the broadcasting feature to compute this for all elements in one go.
    return np.maximum(np.abs(iat - global_min), np.abs(iat - global_max))

def compute_chebyshev_distances_memory_safe_on_frame_len(flow):
    '''
        This is the most optimized way to compute Chebyshev distances, and it works because instead of iterating over each element and compare it with other elements until we find the maximum difference,
        we can simply find the global minimum and maximum of the flow, and then compute the distance of each element to these two extremes. The Chebyshev distance for each element is then the maximum of these two distances.
    '''
    frame_len = np.nan_to_num(flow["frame_len"].values) # ensure no NaN values
    global_min = np.min(frame_len)
    global_max = np.max(frame_len)
    # For 1D Chebyshev, each distance = max(|x - global_min|, |x - global_max|), and numpy uses the broadcasting feature to compute this for all elements in one go.
    return np.maximum(np.abs(frame_len - global_min), np.abs(frame_len - global_max))





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
    print(f"⏱️ Creating {time_window_minutes}-minute windows...")
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
                print(f"⚠️ Error processing pair {futures[f]}: {e}")

    # Merge all processed results into a single DataFrame
    df = pd.concat(time_windowed_flows.values(), keys=time_windowed_flows.keys())
    print(f"✅ Completed {time_window_minutes}-minute flow creation using {max_workers} cores.")
    return df


# # create traffic flows with time window 2 minutes, 4 minutes, and finally 6 minutes
# def create_time_windowed_flows_electra(flows, time_window_minutes):
#     time_windowed_flows = {}
#     print(time_window_minutes)
#     time_delta = pd.Timedelta(minutes=time_window_minutes)
    
#     for pair_id, group in flows.groupby(level=[0]):
#         start_time = group['timestamp'].min()
#         end_time = group['timestamp'].max()
        
#         current_window_start = start_time
#         while current_window_start < end_time:
#             current_window_end = current_window_start + time_delta
#             window_group = group[(group['timestamp'] >= current_window_start) & (group['timestamp'] < current_window_end)]
            
#             if not window_group.empty:
#                 key = (pair_id, current_window_start)
#                 time_windowed_flows[key] = window_group.reset_index(drop=True)
            
#             current_window_start = current_window_end
            
#     # convert the result to a DataFrame with MultiIndex
#     df = pd.concat(time_windowed_flows.values(), keys=time_windowed_flows.keys())
#     return df


def main(): 
    ''' processing_QUT_loaded_dataframe(    '../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/','attacked')
    processing_QUT_loaded_dataframe(    '../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'control')
    '''
    # processing of Electra S7Comm dataset
#    processing_Electra_loaded_dataframe('../../DataSets/electra_s7comm/output/attacked/attacked_data_0.csv', 'attacked')
    processing_Electra_loaded_dataframe('../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv', 'attacked')
    processing_Electra_loaded_dataframe('../../DataSets/electra_s7comm/output/normal/combined_normal.csv', 'normal')

print('start task2 processing')
start = time.time()
main()
end = time.time()

print(f'whole process in {end - start} secs')


