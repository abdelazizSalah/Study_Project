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


def task2e() :
    '''
        We need to compute chebyshev distance between every two raw packets, not two flows
    '''
    






def main(): 
    ''' processing_QUT_loaded_dataframe(    '../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/attacks/','attacked')
    processing_QUT_loaded_dataframe(    '../../DataSets/2017QUT_S7comm/LabelledDataset/output/2017QUT_S7comm/control/', 'control')
    '''
    # processing of Electra S7Comm dataset
    # processing_Electra_loaded_dataframe('../../DataSets/electra_s7comm/output/attacked/combined_attacked.csv', 'attacked')
    processing_Electra_loaded_dataframe('attacked_data_0.csv', 'attacked')
    # processing_Electra_loaded_dataframe('../../DataSets/electra_s7comm/output/normal/combined_normal.csv', 'normal')

print('start task2 processing')
start = time.time()
main()
end = time.time()

print(f'whole process in {end - start} secs')

