# adding Anna's utility functions
import sys, time
sys.path.append('../../Anna_Code')  
from file_helper import *
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed
from utilities import *
import numpy as np



# I noticed that  the combined file I generated is twice the size of the normal electra file, so most probably I should work on the normal electra file directly :)

# Task2 Creating traffic flow for attacker and normal traffic from QUT_S7Comm dataset pcaps
def task2a_preprocessing():
    # processing QUT S7Comm dataset
    QUT_flows_2min_attacked, QUT_flows_4min_attacked, QUT_flows_6min_attacked = task2a_preprocessing_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/', 'attacked')
    QUT_flows_2min_normal, QUT_flows_4min_normal, QUT_flows_6min_normal = task2a_preprocessing_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'control')

    # processing Electra S7Comm dataset
    Electra_flows_2min_attacked, Electra_flows_4min_attacked, Electra_flows_6min_attacked = task2a_preprocessing_Electra('../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv', 'attacked')
    Electra_flows_2min_normal, Electra_flows_4min_normal, Electra_flows_6min_normal = task2a_preprocessing_Electra('../../DataSets/electra_s7comm/output/normal/combined_normal.csv', 'normal')


    # returning all flows
    return QUT_flows_2min_attacked, QUT_flows_4min_attacked, QUT_flows_6min_attacked, QUT_flows_2min_normal, QUT_flows_4min_normal, QUT_flows_6min_normal, Electra_flows_2min_attacked, Electra_flows_4min_attacked, Electra_flows_6min_attacked, Electra_flows_2min_normal, Electra_flows_4min_normal, Electra_flows_6min_normal


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
    iter = 2
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


def task2e(packetsPath = "all_packets.npy") :
    '''
        We need to compute chebyshev distance between every two raw packets, not two flows
    '''
     
    # Generating all packets from all pcaps in a directory
    '''
    all_packets_array = generate_bytes_array_from_packet_list('../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set')
    '''
    # Later load them instantly
    start = time.time()
    all_packets_array = np.load(packetsPath, allow_pickle=True)
    print(f"Loaded {len(all_packets_array)} arrays successfully.")
    print(f'time taken to load all npfile {time.time() - start}\n now creating pairs')

    all_packets =  [] 
    for array in all_packets_array: 
        print(len(array))
        all_packets.extend(arr for arr in array)
    all_packets_array = normalize_packets(all_packets)

    
    # Generate all packet pairs
    start = time.time()
    pairs = list(combinations(all_packets_array[:100], 2))

    print(f'time taken to generate all pairs {time.time() - start} with {len(pairs)}, \n now computing distances')
    start = time.time()
    # Parallel computation for chebyshev distance between packet pairs
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
    return distances

    

def task2f(chebyshev_distance_for_bytes, bins, label):
    bin_edges, counts = manual_histogram(chebyshev_distance_for_bytes, bins=4)
    plt.bar(
        [ (bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(counts)) ],
        counts,
        width=(bin_edges[1] - bin_edges[0]),
        edgecolor='black'
    )
    plt.title(f"Chebyshev Distances {label} Traffic")
    plt.xlabel("Chebyshev Distance (Bytes Level)")
    plt.ylabel("Frequency")
    plt.savefig(f"bytes_{label}.jpg", dpi=300)





def main(): 
    ''' processing_QUT_loaded_dataframe(    '../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/','attacked')
    processing_QUT_loaded_dataframe(    '../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'control')
    '''

    # task2a preprocessing for QUT 
    QUT_Attacked_2mins_flow, QUT_Attacked_4mins_flow, QUT_Attacked_6mins_flow = task2a_preprocessing_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/', 'attacked')
    QUT_Control_2mins_flow, QUT_Control_4mins_flow, QUT_Control_6mins_flow = task2a_preprocessing_QUT('../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'normal')
    
    # task2a preprocessing for Electra
    Electra_Attacked_2mins_flow, Electra_Attacked_4mins_flow, Electra_Attacked_6mins_flow = task2a_preprocessing_Electra('../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv', 'attacked')
    Electra_Normal_2mins_flow, Electra_Normal_4mins_flow, Electra_Normal_6mins_flow = task2a_preprocessing_Electra('../../DataSets/electra_s7comm/output/normal/combined_normal.csv', 'normal')

    # task2b for QUT
    QUT_Attacked_chebyshev_distances_2min,QUT_Attacked_chebyshev_distances_4min,QUT_Attacked_chebyshev_distances_6min,QUT_Attacked_chebyshev_distances_2min_frame_len,QUT_Attacked_chebyshev_distances_4min_frame_len,QUT_Attacked_chebyshev_distances_6min_frame_len = task2b(QUT_Attacked_2mins_flow, QUT_Attacked_4mins_flow, QUT_Attacked_6mins_flow)
    QUT_Control_chebyshev_distances_2min,QUT_Control_chebyshev_distances_4min,QUT_Control_chebyshev_distances_6min,QUT_Control_chebyshev_distances_2min_frame_len,QUT_Control_chebyshev_distances_4min_frame_len,QUT_Control_chebyshev_distances_6min_frame_len = task2b(QUT_Control_2mins_flow, QUT_Control_4mins_flow, QUT_Control_6mins_flow)

    # task2b for Electra
    Electra_Attacked_chebyshev_distances_2min,Electra_Attacked_chebyshev_distances_4min,Electra_Attacked_chebyshev_distances_6min,Electra_Attacked_chebyshev_distances_2min_frame_len,Electra_Attacked_chebyshev_distances_4min_frame_len,Electra_Attacked_chebyshev_distances_6min_frame_len = task2b(Electra_Attacked_2mins_flow, Electra_Attacked_4mins_flow, Electra_Attacked_6mins_flow)
    Electra_Normal_chebyshev_distances_2min,Electra_Normal_chebyshev_distances_4min,Electra_Normal_chebyshev_distances_6min,Electra_Normal_chebyshev_distances_2min_frame_len,Electra_Normal_chebyshev_distances_4min_frame_len,Electra_Normal_chebyshev_distances_6min_frame_len = task2b(Electra_Normal_2mins_flow, Electra_Normal_4mins_flow, Electra_Normal_6mins_flow)

    # task 2c for QUT
    task2c(QUT_Attacked_chebyshev_distances_2min,QUT_Attacked_chebyshev_distances_4min,QUT_Attacked_chebyshev_distances_6min,QUT_Attacked_chebyshev_distances_2min_frame_len,QUT_Attacked_chebyshev_distances_4min_frame_len,QUT_Attacked_chebyshev_distances_6min_frame_len, 'Attacked')
    task2c(QUT_Control_chebyshev_distances_2min,QUT_Control_chebyshev_distances_4min,QUT_Control_chebyshev_distances_6min,QUT_Control_chebyshev_distances_2min_frame_len,QUT_Control_chebyshev_distances_4min_frame_len,QUT_Control_chebyshev_distances_6min_frame_len, 'Normal')

    # task 2c for Electra
    task2c(Electra_Attacked_chebyshev_distances_2min,Electra_Attacked_chebyshev_distances_4min,Electra_Attacked_chebyshev_distances_6min,Electra_Attacked_chebyshev_distances_2min_frame_len,Electra_Attacked_chebyshev_distances_4min_frame_len,Electra_Attacked_chebyshev_distances_6min_frame_len, 'Attacked')
    task2c(Electra_Normal_chebyshev_distances_2min,Electra_Normal_chebyshev_distances_4min,Electra_Normal_chebyshev_distances_6min,Electra_Normal_chebyshev_distances_2min_frame_len,Electra_Normal_chebyshev_distances_4min_frame_len,Electra_Normal_chebyshev_distances_6min_frame_len, 'Normal')


    # task2d: TODO


    # task2e for QUT
    QUT_Normal = task2e('all_packets_normal_qut.npy')
    QUT_Attacked = task2e('all_packets_attacked_qut.npy')
    
    # task2e for Electra
    Electra_Normal = task2e('all_packets_normal_electra.npy')
    Electra_Attacked = task2e('all_packets_attacked_electra.npy')


    # task2f for QUT
    task2f(QUT_Normal, bins=4, label='QUT_Normal')
    task2f(QUT_Attacked, bins=4, label='QUT_Attacked')

    # task2f for Electra
    task2f(Electra_Normal, bins=4, label='Electra_Normal')
    task2f(Electra_Attacked, bins=4, label='Electra_Attacked')



print('start task2 processing')
start = time.time()
main()
end = time.time()

print(f'whole process in {end - start} secs')

