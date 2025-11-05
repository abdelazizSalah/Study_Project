'''
    @Author: Abdelaziz Neamatallah
    @Date: 01.11.25
    @Desc: This script contains the implementation for Task 2 of Sheet 1.
'''

from utilities import *
TESTING = False


# Task2 Creating traffic flow for attacker and normal traffic from QUT_S7Comm dataset pcaps
def task2a_preprocessing():
    # processing QUT S7Comm dataset
    QUT_flows_2min_attacked, QUT_flows_4min_attacked, QUT_flows_6min_attacked = task2a_flows_creation_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/', 'attacked')
    QUT_flows_2min_normal, QUT_flows_4min_normal, QUT_flows_6min_normal = task2a_flows_creation_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'control')

    # processing Electra S7Comm dataset
    Electra_flows_2min_attacked, Electra_flows_4min_attacked, Electra_flows_6min_attacked = task2a_flows_creation_Electra('../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv', 'attacked')
    Electra_flows_2min_normal, Electra_flows_4min_normal, Electra_flows_6min_normal = task2a_flows_creation_Electra('../../DataSets/electra_s7comm/output/normal/combined_normal.csv', 'normal')


    # returning all flows
    return QUT_flows_2min_attacked, QUT_flows_4min_attacked, QUT_flows_6min_attacked, QUT_flows_2min_normal, QUT_flows_4min_normal, QUT_flows_6min_normal, Electra_flows_2min_attacked, Electra_flows_4min_attacked, Electra_flows_6min_attacked, Electra_flows_2min_normal, Electra_flows_4min_normal, Electra_flows_6min_normal


def task2a_flows_creation_QUT(df_path, label):
    # loading the attack and normal datasets from csv files
    loaded_df = multithreading_loading_QUT(df_path)
    # selecting the important features
    if TESTING:
        df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id']][:100] # working on small sample for testing
    else:
        df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id']] # working on small sample for testing

    #df_selected_features = loaded_df[['timestamp', 'src_ip', 'dst_ip', 'app_proto', 'frame_len', 'pair_id']][:1000] # working on small sample for testing

    # drop any packet with undefined app_proto and any packet with no src_ip or dst_ip
    df_selected_features = df_selected_features[df_selected_features['app_proto'] != 'undetected:-1:-1']
    
    # group packets by application protocol
    df_grouped = df_selected_features.groupby(['app_proto', 'pair_id'])

    # selecting interesting features and sorting them by timestamp
    print('before sorting')
    flows = df_grouped.apply(lambda x: x[['timestamp', 'src_ip', 'dst_ip', 'frame_len']].sort_values('timestamp').reset_index(drop=True)) # Reseting the index to get new indicies after sorting to avoid confusion.
    print('after sorting')
    # converting timestamp to datetime
    # flows['timestamp'] = pd.to_datetime(flows['timestamp'], unit='s')

    # Add a column for the direction of the flow, if the src_ip == the first 4 octets of the pair_id, then direction is 0, else 1
    flows['direction'] = flows.apply(determine_direction, axis=1, src_col_name='src_ip', pair_index=1) # src_index is 1 because pair_id is the second level in the MultiIndex
    
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

    # if the file .paraquet exists, load it directly
    # if os.path.exists(label + '.parquet'):
    #     print(f'loading preprocessed parquet file for {label}')
    #     now = time.time()
    #     flows = pd.read_parquet(label + '.parquet')
    #     print(f'loaded parquet file in {time.time() - now} secs')
 	# # There is a problem in loading from .parquet.

    # else: 
    if True:
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
        print(f'writing csv file {label}0.csv')
        now = time.time()
        flows.to_parquet(label + '.parquet', index=False)  

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


def task2d_with_no_parallelization(flows):
        
    # --- Hyperparameter tuning (grid search) ---
    # flows = [flows_control_2min, flows_control_4min, flows_control_6min]
    if TESTING:
        perplexities = [10]
        learning_rates = [100]
        flows = flows[:2]
    else:
        perplexities = [10, 30, 50]
        learning_rates = [100, 200, 500]
        
    for i, flow in enumerate(flows):
        best_score = float('inf')
        best_tsne = None
        X = flow[['frame_len', 'iat']].values
        X = StandardScaler().fit_transform(X)  # Standardize features
        X = np.nan_to_num(X)  # Ensure no NaN values
        for p in perplexities:
            for lr in learning_rates:
                tsne = TSNE(n_components=2, perplexity=p, learning_rate=lr, random_state=42)
                embedding = tsne.fit_transform(X)
                kl_div = tsne.kl_divergence_  # lower is better
                if kl_div < best_score:
                    best_score = kl_div
                    best_tsne = embedding
        
        # --- Plot best embedding ---
        plt.figure(figsize=(6,5))
        plt.scatter(best_tsne[:,0], best_tsne[:,1], s=10, alpha=0.7)
        plt.title(f"t-SNE (control) - {(i + 1) * 2} min window\nBest KL: {best_score:.4f}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.tight_layout()
        plt.savefig(f"tsne_control_{(i + 1) * 2}min.png", dpi=300)
        plt.close()
    return True


def tsne_worker(args):
    return compute_best_tsne(*args)

def task2d(flows):
    perplexities, learning_rates = [10,30,50], [100,200,500]
    workers = os.cpu_count() or 1
    args_list = [(flow, perplexities, learning_rates, i) for i, flow in enumerate(flows)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(tsne_worker, args_list))
    print("Best KL divergences per flow:", results)
    return True



def task2e(packetsPathNpy = "all_packets.npy", packetListPathPcap='../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set', label='control') :
    '''
        We need to compute chebyshev distance between every two raw packets, not two flows
    '''
     
    
    if not os.path.exists(packetsPathNpy):
        # Generating all packets from all pcaps in a directory
        print('generating bytes')
        all_packets_array = generate_bytes_array_from_packet_list(packetListPathPcap)
    else: 
        start = time.time()
        all_packets_array = np.load(packetsPathNpy, allow_pickle=True)
        print(f"Loaded {len(all_packets_array)} arrays successfully.")
        print(f'time taken to load all npfile {time.time() - start}\n now creating pairs')

    all_packets =  [] 
    for array in all_packets_array: 
        print(len(array))
        all_packets.extend(arr for arr in array)
    all_packets_array = normalize_packets(all_packets)

    
    # Generate all packet pairs
    now = time.time()
    if TESTING:
        pairs = list(combinations(all_packets_array[:100], 2)) # This 100 should be changed after I ask Dr.Asya
    else: 
        pairs = list(combinations(all_packets_array, 2)) 

    print(f'time taken to generate all pairs {time.time() - now} with {len(pairs)}, \n now computing distances')
    now = time.time()
    # Parallel computation for chebyshev distance between packet pairs
    maxWorkers = os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=maxWorkers) as executor:
        distances = list(executor.map(compute_pair_distance, pairs))

    print(f'time taken to compute chebychev distance files {time.time() - now}')
    now = time.time()
    print(f"Computed {len(distances)} Chebyshev distances in parallel.\n now saving to file the distances")

    # Save to file (binary .npy for fast I/O)
    output_path = f"chebyshev_distances_{label}.npy"
    np.save(output_path, np.array(distances, dtype=np.float32))
    print(f"Saved distances to {output_path}")

    print(f'time taken to write chebyshev distance to file {time.time() - now}')
    #start = time.time()
    print(len(distances))
    print(distances[:100])
    return distances


def task2f(chebyshev_distance_for_bytes, bins, label):
    bin_edges, counts = manual_histogram(chebyshev_distance_for_bytes, bins=bins)
    plot_histogram(bin_edges, counts, f"Chebyshev Distances {label} Traffic", "Chebyshev Distance (Bytes Level)", "Frequency", f"bytes_{label}.jpg")
    



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
    task2d_gpu([QUT_Control_2mins_flow, QUT_Control_4mins_flow, QUT_Control_6mins_flow])
    task2d_gpu([QUT_Attacked_2mins_flow, QUT_Attacked_4mins_flow, QUT_Attacked_6mins_flow])
   
  
 
    ## task2d for Electra
    print('processing task 2d E')
    task2d_gpu([Electra_Normal_2mins_flow, Electra_Normal_4mins_flow, Electra_Normal_6mins_flow])
    task2d_gpu([Electra_Attacked_2mins_flow, Electra_Attacked_4mins_flow, Electra_Attacked_6mins_flow])

    # task2e for QUT only
    print('processing task 2e Q')
    QUT_Normal = task2e('all_packets_normal_qut.npy', '../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set',label='QUT_control_taskE')
    QUT_Attacked = task2e('all_packets_attacked_qut.npy', '../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks', label='QUT_attacked_taskE')
   
    
    # task2f for QUT only
    print('processing task 2f Q ')
    task2f(QUT_Normal, bins=4, label='QUT_Normal_taskF')
    task2f(QUT_Attacked, bins=4, label='QUT_Attacked_taskF')



print('start task2 processing')
start = time.time()
main()
end = time.time()

print(f'whole process in {end - start} secs')



