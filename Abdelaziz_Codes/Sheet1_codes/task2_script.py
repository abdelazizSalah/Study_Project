'''
    @Author: Abdelaziz Neamatallah
    @Date: 01.11.25
    @Desc: This script contains the implementation for Task 2 of Sheet 1.
'''

from utilities import *
import multiprocessing as mp
TESTING = True



def task2a_flows_creation_QUT(df_path, label):
    # loading the attack and normal datasets from csv files
    loaded_df = multithreading_loading_QUT(df_path)
    loaded_df['attack_label'] = loaded_df['filename'].apply(lambda x: x.split('.')[1] if len(x.split('.')) > 2 else x.split('.')[0])
    # selecting the important features
    if TESTING:
        df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id', 'attack_label']][:100] # working on small sample for testing
    else:
        df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id', 'attack_label']] # working on small sample for testing

    #df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id']][:1000] # working on small sample for testing

    # drop any packet with undefined app_proto and any packet with no src_ip or dst_ip
    # df_selected_features = df_selected_features[df_selected_features['app_proto'] != 'undetected:-1:-1']
    
    # group packets by application protocol
    df_grouped = df_selected_features.groupby(['pair_id','app_proto'])

    # selecting interesting features and sorting them by timestamp
    print('before sorting')
    flows = df_grouped.apply(lambda x: x[['timestamp', 'src_ip', 'dst_ip', 'frame_len', 'attack_label']].sort_values('timestamp').reset_index(drop=True)) # Reseting the index to get new indicies after sorting to avoid confusion.
    print('after sorting')
    # converting timestamp to datetime
    # flows['timestamp'] = pd.to_datetime(flows['timestamp'], unit='s')

    # Add a column for the direction of the flow, if the src_ip == the first 4 octets of the pair_id, then direction is 0, else 1
    flows['direction'] = flows.apply(determine_direction, axis=1, src_col_name='src_ip', pair_index=0) # src_index is 1 because pair_id is the second level in the MultiIndex
    
    # adding inter-arrival time between packets in the same flow
#    flows['iat'] = flows.groupby(level=[0,1])['timestamp'].diff().dt.total_seconds() # levels here refers to the index of the groups which are the protocol and pair_id
    flows['iat'] = flows.groupby(level=[0,1])['timestamp'].diff() # levels here refers to the index of the groups which are the protocol and pair_id

    print(f'data head before flows creation {flows.head()}')
    print(flows.columns)
    # creating time_widnowed_flows with 2, 4, and 6 mins
    flows_2min = create_time_windowed_flows(flows, 2)
    flows_4min = create_time_windowed_flows(flows, 4)
    flows_6min = create_time_windowed_flows(flows, 6)
    # printing thier size
    print(f" data from {label} \n flows_2min len: {len(flows_2min)},  flows_4min len: {len(flows_4min)},  flows_6min len: {len(flows_6min)}.")

    # printing number of flows per protocol
    print(f"{label} number of flows per protocol\n 2min: {flows_2min.index.get_level_values(0).value_counts()}\n 4min: {flows_4min.index.get_level_values(0).value_counts()}\n 6min: {flows_6min.index.get_level_values(0).value_counts()}") # this is the number of flows per application protocol, and if we summed them up, it should be equal to the total number of flows we got earlier.

    return flows_2min, flows_4min, flows_6min

def task2a_flows_creation_Electra(df_path, label):

    # Time,smac,dmac,sip,dip,request,fc,error,address,data,label -> headers
    # loading the attack and normal datasets from csv files
    print('task2a starting')
    start = time.time()
    # this is the combined csv file, after merging and preprocessing it and selecting interesting features only. 
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


    # determine the direction determine direction, and the argument is sip
    flows['direction'] = flows.apply(determine_direction, axis=1, src_col_name='sip', pair_index=0) 
    print('after the direction')
    # converting timestamp to datetimestamp
    # this should not be changed to date_time, because it is relative timestamp
#        flows['timestamp'] = pd.to_datetime(flows['timestamp'], unit='s') # not sure if this will be a good idea.
    print('after timestamp conversion')
    # adding inter-arrival timestamp between packets in the same flow
#        flows['iat'] = flows.groupby(level=[0])['timestamp'].diff().dt.total_seconds() # levels here refers to the index of the groups
    flows['iat'] = flows.groupby(level=[0])['timestamp'].diff() # levels here refers to the index of the groups
    print('adding iat')
    print('starting creating flows 2 mins')
    
    # writing the file to save time instead of recomputing everything again
    print(f'writing csv file {label}_electra.csv')
    now = time.time()
    flows.to_parquet(label + '_electra.parquet', index=False)  

    print(f'processing time to write to the disk: {time.time() - now}') 
    print(f'data head before flows creation {flows.head()}')
    print(flows.columns)
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
    chebyshev_distances_2min = compute_chebyshev_distances_on_iat_optimized(flows_2min)
    chebyshev_distances_4min = compute_chebyshev_distances_on_iat_optimized(flows_4min)
    chebyshev_distances_6min = compute_chebyshev_distances_on_iat_optimized(flows_6min)

    # print on frame_len column
    chebyshev_distances_2min_frame_len = compute_chebyshev_distances_on_frame_len_optimized(flows_2min)
    chebyshev_distances_4min_frame_len = compute_chebyshev_distances_on_frame_len_optimized(flows_4min)
    chebyshev_distances_6min_frame_len = compute_chebyshev_distances_on_frame_len_optimized(flows_6min)

    return chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len

def task2c(chebyshev_distances_2min,chebyshev_distances_4min,chebyshev_distances_6min,chebyshev_distances_2min_frame_len,chebyshev_distances_4min_frame_len,chebyshev_distances_6min_frame_len, label):
    print('start processing task2c')
    iter = 2
    for data in [chebyshev_distances_2min, chebyshev_distances_4min, chebyshev_distances_6min]:
        bin_edges, counts = manual_histogram(data, bins=4)
        plot_histogram(bin_edges, counts, f"Chebyshev Distances {label} Traffic", "Chebyshev Distance (IAT)", "Frequency", f"iat_{iter}_{label}.jpg")
        iter+=2
 #       plt.show()
    iter = 2
    for data in [chebyshev_distances_2min_frame_len, chebyshev_distances_4min_frame_len, chebyshev_distances_6min_frame_len]:
        bin_edges, counts = manual_histogram(data, bins=4)
        plot_histogram(bin_edges, counts, f"Chebyshev Distances {label} Traffic", "Chebyshev Distance (Frame Length)", "Frequency", f"frameLen_{iter}_{label}.jpg")
        iter+= 2
#        plt.show()


def task2d_QUT(flows, best_perplexity = 50, label='QUT_attacked'):
    # running the model on the whole data with the best parameters found
    X = flows[['frame_len', 'iat', 'direction']].values
    X = StandardScaler().fit_transform(X)  # Standardize features
    X = np.nan_to_num(X)  # Ensure no NaN values
    embedding = TSNE(n_components=2, perplexity=best_perplexity, metric='chebyshev').fit_transform(X)

    print('finished')
    print(embedding)
    # --- Plot best embedding ---

    plt.figure(figsize=(6,5))
    # Assign the color based on the attack_label, green for hmi, blue for master, and red for others
    attack_labels = flows['attack_label']
    color_map = {'master': 'green', 'hmi': 'blue'}
    colors = attack_labels.map(color_map).fillna('red')

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color)
                for label, color in color_map.items()]
    # add to handles red for attacks 
    handles.append(plt.Line2D([0], [0], marker='o', color='w', label='other attacks', markerfacecolor='red'))

    plt.legend(handles=handles, title="Attack Type")
    plt.scatter(embedding[:,0], embedding[:,1],  s=10, alpha=0.7, c=colors)
    plt.title(f"t-SNE (attack)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"tsne_fulldata_{label}.png", dpi=300)
    # I left it the whole day yesterday running, and it did not finish... 

def task2d_Electra(Electra_Attacked_2mins_flow, Electra_Normal_2mins_flow, label='Electra_2min_attack_normal'):     
    # combine the flows 2 min control but add to them label = normal with the 2 min attack with label attack 
    Electra_Attacked_2mins_flow['label']= 'attack'
    Electra_Normal_2mins_flow['label']= 'normal'
    flows_2min = pd.concat([Electra_Attacked_2mins_flow, Electra_Normal_2mins_flow], ignore_index=True)
    
    # running the model on the whole data with the best parameters found
    X = flows_2min[['frame_len', 'iat', 'direction']].values
    X = StandardScaler().fit_transform(X)  # Standardize features
    X = np.nan_to_num(X)  # Ensure no NaN values
    embedding = TSNE(n_components=2, perplexity=50, metric='chebyshev').fit_transform(X)

    print('finished')
    # --- Plot best embedding ---

    plt.figure(figsize=(6,5))
    # Assign the color based on the attack_label, green for hmi, blue for master, and red for others
    attack_labels = flows_2min['label']
    color_map = {'normal': 'green', 'attack': 'red'}
    colors = attack_labels.map(color_map).fillna('red')

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color)
                for label, color in color_map.items()]
    # add to handles red for attacks 
    plt.legend(handles=handles, title="Attack Type")
    plt.scatter(embedding[:,0], embedding[:,1],  s=10, alpha=0.7, c=colors)
    plt.title(f"t-SNE ({label} flows: normal + attack)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(f"tsne_fulldata_{label}.png", dpi=300)



# def parameter_tuning(flow, perplexities, learning_rates):
#     best_score = float('inf')
#     best_perplexity, best_lr = 0, 0
#     best_tsne = None
#     for p in perplexities:
#         for lr in learning_rates:
#             tsne = TSNE(n_components=2, perplexity=p, learning_rate=lr, random_state=42, metric='chebyshev') # the output will be strange because we are using chebyshev distance, which does not make sense in our context, however we can use eucliedean distance for better representation.
#             embedding = tsne.fit_transform(flow)
#             kl_div = tsne.kl_divergence_  # lower is better
#             if kl_div < best_score:
#                 best_score = kl_div
#                 best_tsne = embedding
#                 best_perplexity, best_lr = p, lr
#     return  best_perplexity, best_lr, best_score, best_tsne


# def task2d_with_no_parallelization(flows, label):
        
#     # --- Hyperparameter tuning (grid search) ---
#     # flows = [flows_control_2min, flows_control_4min, flows_control_6min]
#     if TESTING:
#         perplexities = [10]
#         learning_rates = [100]
#         flows = flows[:2][:100]
#     else:
#         perplexities = [i for i in range(5,50)]
#         learning_rates = [i for i in range (10,1000)]

#         # use the best parameters to compute t-SNE embeddings for all flows
#         for i, flow in enumerate(flows):
#             X = flow[['frame_len', 'iat', 'direction']].values
#             X = StandardScaler().fit_transform(X)  # Standardize features
#             X = np.nan_to_num(X)  # Ensure no NaN values
#             best_perplexity, best_lr, best_score, _ = parameter_tuning(X[:int(len(X) * 0.1)], perplexities, learning_rates)
           
#             # running the model again but with the best parameters on the full data
#             tsne = TSNE(n_components=2, perplexity=best_perplexity, learning_rate=best_lr, random_state=42, metric='chebyshev')
#             embedding = tsne.fit_transform(X)
#             # --- Plot best embedding ---
#             # Assign the color based on the attack_label, green for hmi, blue for master, and red for others
                        
#             color_map = {'master': 'green', 'hmi': 'blue'}
#             handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color)
#                         for label, color in color_map.items()]
#             if label != 'control':
#                 attack_labels = flow['attack_label']
#                 colors = attack_labels.map(color_map).fillna('red')
#                 # add to handles red for attacks 
#                 handles.append(plt.Line2D([0], [0], marker='o', color='w', label='other attacks', markerfacecolor='red'))

#             plt.legend(handles=handles, title="Attack Type")
#             plt.figure(figsize=(6,5))
#             plt.scatter(embedding[:,0], embedding[:,1], s=10, alpha=0.7, c=colors)
#             plt.title(f"t-SNE ({label}) - {(i + 1) * 2} min window\nBest KL: {best_score:.4f}")
#             plt.xlabel("t-SNE Dimension 1")
#             plt.ylabel("t-SNE Dimension 2")
#             plt.tight_layout()
#             plt.savefig(f"tsne_{label}_{(i + 1) * 2}min.png", dpi=300)
#             plt.close()
#     return True


# def tsne_worker(args):
#     return compute_best_tsne(*args)

# # This should be modified. 
# def task2d(flows):
#     perplexities, learning_rates = [p for p in range(5,50)], [lr for lr in range(10,1000)]
#     workers = os.cpu_count() or 1
#     args_list = [(flow, perplexities, learning_rates, i) for i, flow in enumerate(flows)]
#     with ProcessPoolExecutor(max_workers=workers) as executor:
#         results = list(executor.map(tsne_worker, args_list))
#     print("Best KL divergences per flow:", results)
#     return True

# ##############################################3


def pad_packets_to_max(packets):
    """Convert packets (bytes/bytearray) into uniform 2D numpy array padded to max length."""
    max_len = max(len(np.frombuffer(p, dtype=np.uint8)) for p in packets)
    print(f"[INFO] Max packet length = {max_len} bytes")

    padded = np.zeros((len(packets), max_len), dtype=np.uint8)
    for i, p in enumerate(packets):
        arr = np.frombuffer(p, dtype=np.uint8)
        padded[i, :len(arr)] = arr
    return padded


def compute_block(args):
    """Worker: compute Chebyshev distances for a pair of blocks (i,j)."""
    i, j, block_a, block_b, label, block_id = args
    print(f"  -> block {block_id}: ({i}:{i+len(block_a)}) x ({j}:{j+len(block_b)})")

    # Compute Chebyshev distances using broadcasting
    diffs = np.abs(block_a[:, None, :] - block_b[None, :, :])
    cheb = np.max(diffs, axis=2)

    # Keep only upper triangle if same block
    if i == j:
        mask = np.triu(np.ones_like(cheb, dtype=bool), k=1)
        cheb = cheb[mask]

    out = f"chebyshev_blocks/{label}_block_{block_id}.npy"
    

    print(f'saving the block {block_id}')
    np.save(out, cheb.astype(np.float32))
    return out


def compute_chebyshev_blockwise_parallel(all_packets_array, label, block_size=1000):
    """Parallel computation of Chebyshev distances blockwise."""
    os.makedirs("chebyshev_blocks", exist_ok=True)
    print('padding')
    # pad first, ensure rectangular numeric array
    all_packets_array = pad_packets_to_max(all_packets_array)

    n = len(all_packets_array)
    
    jobs, block_id = [], 0

    # Build job list safely within bounds
    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(i, n, block_size):
            j_end = min(j + block_size, n)
            block_a = all_packets_array[i:i_end]
            block_b = all_packets_array[j:j_end]
            jobs.append((i, j, block_a, block_b, label, block_id))
            block_id += 1

    start = time.time()
    print(f"[INFO] Launching {len(jobs)} block tasks with {os.cpu_count()} workers")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(compute_block, job) for job in jobs]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"[WARN] Block failed: {e}")

    print(f"[INFO] Done in {(time.time()-start)/60:.2f} min; "
          f"results saved in 'chebyshev_blocks/'")
    

def task2e_optimized(packetsPathNpy = "all_packets.npy", packetListPathPcap='../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set', label='control') :
    '''
        We need to compute chebyshev distance between every two raw packets, not two flows
    '''
     
    
    if not os.path.exists(packetsPathNpy):
        # Generating all packets from all pcaps in a directory
        print('generating bytes')
        all_packets_array = generate_bytes_array_from_packet_list(packetListPathPcap, label=label)
    else: 
        start = time.time()
        all_packets_array = np.load(packetsPathNpy, allow_pickle=True)# allow pickle is a parameter in numpy which allow storing and loading different formats, and I read that it suggested to be true to make it faster. 
        print(f"Loaded {len(all_packets_array)} arrays successfully.")
        print(f'time taken to load all npfile {time.time() - start}\n now creating pairs')
    
    all_packets =  [] 
    for array in all_packets_array: 
        print('extending packets')
        all_packets.extend(arr for arr in array)
    print(f'total packets length {len(all_packets)} before normailizing')
    print('now normalizing packets')
    all_packets_array = normalize_packets(all_packets)
    print(f'total packets length {len(all_packets_array)} after normalizing')

    
    # Generate all packet pairs
    if TESTING:
        all_packets_array = all_packets_array[:100]  # for testing, use a smaller subset
    print('computing all distances now')
    return compute_chebyshev_blockwise_parallel(all_packets_array, label)
    


def task2f(chebyshev_distance_for_bytes, bins, label):
    bin_edges, counts = manual_histogram(chebyshev_distance_for_bytes, bins=bins)
    plot_histogram(bin_edges, counts, f"Chebyshev Distances {label} Traffic", "Chebyshev Distance (Bytes Level)", "Frequency", f"bytes_{label}.jpg")
    
    
def worker(file_path, shared_hist, lock):
    print(f"[+] Processing {file_path}")
    data = np.load(file_path)
    local_hist, _ = np.histogram(data, bins=5, range=(0, 255))
    with lock:
        for i in range(5):
            shared_hist[i] += int(local_hist[i])
    print(f"[✓] Finished {file_path}")



def task2fMultiProcessing(label='QUT_attacked_taskE', folder = "chebyshev_blocks"):
     
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f"{label}" in f]
        if TESTING:
            files = files[:2]

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

        # Generate correct bin edges and labels
        bin_edges = np.linspace(0, 255, 6)  # 6 edges → 5 bins
        bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])}" for i in range(5)]

        plt.figure(figsize=(8, 4))
        plt.bar(np.arange(5), global_hist, color='steelblue')
        plt.xticks(np.arange(5), bin_labels, rotation=30)
        plt.title(f'Global Histogram (5 bins) of Chebyshev Distances – {label}')
        plt.xlabel('Distance Range')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'global_histogram_{label}_5bins.png')
        plt.close()

        print(f"[✓] Histogram saved as 'global_histogram_{label}_5bins.png'")



def main(): 
    
    # task2a preprocessing for QUT 
    print('processing task 2a Q')
    QUT_Attacked_2mins_flow, QUT_Attacked_4mins_flow, QUT_Attacked_6mins_flow = task2a_flows_creation_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/', 'QUT_attacked')
    QUT_Control_2mins_flow, QUT_Control_4mins_flow, QUT_Control_6mins_flow = task2a_flows_creation_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'QUT_normal')
    
    # task2a preprocessing for Electra
    print('processing task 2a E')

    Electra_Attacked_2mins_flow, Electra_Attacked_4mins_flow, Electra_Attacked_6mins_flow = task2a_flows_creation_Electra('../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv', 'Electra_attacked')
    Electra_Normal_2mins_flow, Electra_Normal_4mins_flow, Electra_Normal_6mins_flow = task2a_flows_creation_Electra('../../DataSets/electra_s7comm/output/normal/combined_normal.csv', 'Electra_normal')

    # task2b for QUT
    print('processing task 2b Q')
    QUT_Attacked_chebyshev_distances_2min,QUT_Attacked_chebyshev_distances_4min,QUT_Attacked_chebyshev_distances_6min,QUT_Attacked_chebyshev_distances_2min_frame_len,QUT_Attacked_chebyshev_distances_4min_frame_len,QUT_Attacked_chebyshev_distances_6min_frame_len = task2b(QUT_Attacked_2mins_flow, QUT_Attacked_4mins_flow, QUT_Attacked_6mins_flow)
    QUT_Control_chebyshev_distances_2min,QUT_Control_chebyshev_distances_4min,QUT_Control_chebyshev_distances_6min,QUT_Control_chebyshev_distances_2min_frame_len,QUT_Control_chebyshev_distances_4min_frame_len,QUT_Control_chebyshev_distances_6min_frame_len = task2b(QUT_Control_2mins_flow, QUT_Control_4mins_flow, QUT_Control_6mins_flow)

    # task2b for Electra
    print('processing task 2b E')
    Electra_Attacked_chebyshev_distances_2min,Electra_Attacked_chebyshev_distances_4min,Electra_Attacked_chebyshev_distances_6min,Electra_Attacked_chebyshev_distances_2min_frame_len,Electra_Attacked_chebyshev_distances_4min_frame_len,Electra_Attacked_chebyshev_distances_6min_frame_len = task2b(Electra_Attacked_2mins_flow, Electra_Attacked_4mins_flow, Electra_Attacked_6mins_flow)
    Electra_Normal_chebyshev_distances_2min,Electra_Normal_chebyshev_distances_4min,Electra_Normal_chebyshev_distances_6min,Electra_Normal_chebyshev_distances_2min_frame_len,Electra_Normal_chebyshev_distances_4min_frame_len,Electra_Normal_chebyshev_distances_6min_frame_len = task2b(Electra_Normal_2mins_flow, Electra_Normal_4mins_flow, Electra_Normal_6mins_flow)

    # task 2c for QUT
    print('processing task 2c Q')
    task2c(QUT_Attacked_chebyshev_distances_2min,QUT_Attacked_chebyshev_distances_4min,QUT_Attacked_chebyshev_distances_6min,QUT_Attacked_chebyshev_distances_2min_frame_len,QUT_Attacked_chebyshev_distances_4min_frame_len,QUT_Attacked_chebyshev_distances_6min_frame_len, 'QUT_Attacked_taskC')
    task2c(QUT_Control_chebyshev_distances_2min,QUT_Control_chebyshev_distances_4min,QUT_Control_chebyshev_distances_6min,QUT_Control_chebyshev_distances_2min_frame_len,QUT_Control_chebyshev_distances_4min_frame_len,QUT_Control_chebyshev_distances_6min_frame_len, 'QUT_Normal_taskC')

    # task 2c for Electra
    print('processing task 2c E')
    task2c(Electra_Attacked_chebyshev_distances_2min,Electra_Attacked_chebyshev_distances_4min,Electra_Attacked_chebyshev_distances_6min,Electra_Attacked_chebyshev_distances_2min_frame_len,Electra_Attacked_chebyshev_distances_4min_frame_len,Electra_Attacked_chebyshev_distances_6min_frame_len, 'Electra_Attacked_taskC')
    task2c(Electra_Normal_chebyshev_distances_2min,Electra_Normal_chebyshev_distances_4min,Electra_Normal_chebyshev_distances_6min,Electra_Normal_chebyshev_distances_2min_frame_len,Electra_Normal_chebyshev_distances_4min_frame_len,Electra_Normal_chebyshev_distances_6min_frame_len, 'Electra_Normal_taskC')
    
    # task2d for QUT
    print('processing task 2d Q')
    task2d_QUT(QUT_Attacked_2mins_flow, label='QUT_2mins_attacked')  
    if not TESTING: 
        task2d_QUT(QUT_Attacked_4mins_flow, label='QUT_4mins_attacked')  
        task2d_QUT(QUT_Attacked_6mins_flow, label='QUT_6mins_attacked')  
        
        task2d_QUT(QUT_Control_2mins_flow, label='QUT_2mins_Control')  
        task2d_QUT(QUT_Control_4mins_flow, label='QUT_4mins_Control')  
        task2d_QUT(QUT_Control_6mins_flow, label='QUT_6mins_Control')  
 
    ## task2d for Electra
    print('processing task 2d E')
    # task2d([Electra_Normal_2mins_flow, Electra_Normal_4mins_flow, Electra_Normal_6mins_flow], 'Electra_Normal')
    # task2d([Electra_Attacked_2mins_flow, Electra_Attacked_4mins_flow, Electra_Attacked_6mins_flow], 'Electra_Attacked')
    task2d_Electra(Electra_Attacked_2mins_flow, Electra_Normal_2mins_flow, label='Electra_2min_attack_normal')
    
    if not TESTING:
        task2d_Electra(Electra_Attacked_4mins_flow, Electra_Normal_4mins_flow, label='Electra_4min_attack_normal')
        task2d_Electra(Electra_Attacked_6mins_flow, Electra_Normal_6mins_flow, label='Electra_6min_attack_normal')
    
   
    # task2e for QUT only
    print('processing task 2e Q')
    task2e_optimized('all_packets_QUT_control_taskE.npy', '../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set',label='QUT_control_taskE')
    print('processing task 2f Q')
    task2fMultiProcessing(label='QUT_control_taskE')

    if not TESTING:
        task2e_optimized('all_packets_QUT_attacked_taskE.npy', '../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks', label='QUT_attacked_taskE')
        task2fMultiProcessing(label='QUT_attacked_taskE')


print('start task2 processing')
start = time.time()
main()
end = time.time()

print(f'whole process in {end - start} secs')



