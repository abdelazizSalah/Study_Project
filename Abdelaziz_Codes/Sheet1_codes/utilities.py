

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
    counts = [0] * bins # create a list for counts and initialze it with zeros. 

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

