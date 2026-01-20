
# import pytorch libraries
TESTING = True
import numpy as np
import torch
import argparse
import os
import sys
import numpy as np
from collections import Counter
import math


# Phase 1: Data Prepration

def load_phase1_saved_data(training_input_path = './data/training_data.npy',validation_input_path = './data/validation_data.npy',testing_input_path = './data/testing_data.npy', training_labels_path = './data/training_labels.npy', validation_labels_path = './data/validation_labels.npy', test_labels_path = './data/testing_labels.npy'):
    # load training, validation, testing data from .npy
    training_data = torch.from_numpy(np.load(training_input_path))
    validation_data = torch.from_numpy(np.load(validation_input_path))
    testing_data = torch.from_numpy(np.load(testing_input_path))

    # load training, validation, testing labels from .npy
    training_labels = np.load(training_labels_path, allow_pickle=True)
    validation_labels = np.load(validation_labels_path, allow_pickle=True)
    test_labels = np.load(test_labels_path, allow_pickle=True)

    print('loaded sucessfully after modification')
    return training_data, validation_data, testing_data, training_labels, validation_labels, test_labels


def phase1_read_arguments():
    '''
        This function reads the arguments from command line
        - M: number of bytes per sample
        - use_stats: whether to use statistical features or not
        - p: number of physical readings (for RE mode)
    '''
    parser = argparse.ArgumentParser(
        description="ResNet-based ICS traffic classifier (Task 1)"
    )

    # Number of bytes per sample
    parser.add_argument(
        "-M",
        type=int,
        required=True,
        help="Number of bytes per packet/sample (trim or pad to M bytes)"
    )

    # Use statistical features or not
    parser.add_argument(
        "--use_stats",
        action="store_true",
        help="Include 5 statistical features in addition to raw bytes"
    )

    # Physical readings count (for RE mode)
    parser.add_argument(
        "--p",
        type=int,
        choices=[5, 10, 15],
        default=None,
        help="Number of physical readings for reverse-engineered mode"
    )
    args = parser.parse_args()
    return args.M, args.use_stats, args.p

def phase1_getting_data(n,m, use_stats):
    '''
        This function is responsible for checking if the data and labels exist or not, and if not, to generate them
        Input:
            - n: number of bytes per packet
            - m: number of packets per sample
        Output:
            - training_data: tensor of shape (num_training_samples, 1, m, n)
            - validation_data: tensor of shape (num_validation_samples, 1, m, n)
            - testing_data: tensor of shape (num_testing_samples, 1, m, n)
            - training_labels: numpy array of shape (num_training_samples,)
            - validation_labels: numpy array of shape (num_validation_samples,)
            - test_labels: numpy array of shape (num_testing_samples,)
    '''


    # check if data and labels already exist.
    training_input_path = f'./data/training_data_n_{n}_m_{m}.npy'
    validation_input_path = f'./data/validation_data_n_{n}_m_{m}.npy'
    testing_input_path = f'./data/testing_data_n_{n}_m_{m}.npy'

    training_labels_path = f'./data/training_labels_n_{n}_m_{m}.npy'
    validation_labels_path = f'./data/validation_labels_n_{n}_m_{m}.npy'
    testing_labels_path = f'./data/testing_labels_n_{n}_m_{m}.npy'
    print('Checking for data files:'
          f'\n - Training data: {training_input_path}'
          f'\n - Validation data: {validation_input_path}'
          f'\n - Testing data: {testing_input_path}'
          )
    Data_Files_Exist = os.path.exists(f'./data/training_data_n_{n}_m_{m}.npy') and os.path.exists(f'./data/validation_data_n_{n}_m_{m}.npy') and os.path.exists(f'./data/testing_data_n_{n}_m_{m}.npy') and os.path.exists(f'./data/training_labels_n_{n}_m_{m}.npy') and os.path.exists(f'./data/validation_labels_n_{n}_m_{m}.npy') and os.path.exists(f'./data/testing_labels_n_{n}_m_{m}.npy')
    if Data_Files_Exist:
        # load them
        print('[*] Phase 1: Data files found. Loading...')
        training_data, validation_data, testing_data, training_labels, validation_labels, test_labels = load_phase1_saved_data(
            training_input_path=training_input_path, 
            validation_input_path=validation_input_path, 
            testing_input_path=testing_input_path, 
            training_labels_path=training_labels_path,
            validation_labels_path=validation_labels_path,
            test_labels_path=testing_labels_path
        )
        # print unique labels for training, validation, and testing
        print(f"[*] Training labels unique values: {np.unique(training_labels)}")
        print(f"[*] Validation labels unique values: {np.unique(validation_labels)}")
        print(f"[*] Testing labels unique values: {np.unique(test_labels)}")
        
        print('[*] Phase 1: Data files found. Loaded successfully.')

    else: 
        # Load and preprocess data
        print('[*] Phase 1: Data files not found. Preparing data...')
        normal_data, attack_data, features_normal, features_attack = phase1_data_prepration(n, use_stats)
        print(f"[*] Total normal samples: {len(normal_data)}")
        print(f"[*] Total attack samples: {len(attack_data)}")
        print('======================================================')
        print(f'[*] normal_features samples: {len(features_normal)}')
        print(f'[*] attack_features samples: {len(features_attack)}')
        print('======================================================')


        # Convert data to tensors
        normalData = prepare_tensors(normal_data, n, m=m)  
        attackData = prepare_tensors(attack_data, n, m=m)  

        print('[*] Data converted to tensors successfully.\n --------------------------------------- \n splitting data into training, validation, testing sets...')
        training_data, validation_data, testing_data, training_labels, validation_labels, test_labels = phase1_dataset_splitting(
            # converting normal data to tensor
            normal_data = normalData,

            # converting attack data to tensor
            attack_data = attackData, 
        )
        # Save training, validation, testing data into .npy
        # make data folders if not exist
        if not os.path.exists('./data'):
            os.makedirs('./data')
        np.save(f'./data/training_data_n_{n}_m_{m}.npy', training_data.numpy())
        np.save(f'./data/validation_data_n_{n}_m_{m}.npy', validation_data.numpy())
        np.save(f'./data/testing_data_n_{n}_m_{m}.npy', testing_data.numpy())
        print('Saved training, validation, testing data into .npy files.')

        # Save training, validation, testing labels into .npy
        np.save(f'./data/training_labels_n_{n}_m_{m}.npy', training_labels)
        np.save(f'./data/validation_labels_n_{n}_m_{m}.npy', validation_labels)
        np.save(f'./data/testing_labels_n_{n}_m_{m}.npy', test_labels)
        print('Saved training, validation, testing labels into .npy files.')



        # print unique labels for training, validation, and testing
        print(f"[*] Training labels unique values: {np.unique(training_labels)}")
        print(f"[*] Validation labels unique values: {np.unique(validation_labels)}")
        print(f"[*] Testing labels unique values: {np.unique(test_labels)}")
    return training_data, validation_data, testing_data, training_labels, validation_labels, test_labels


def phase1_dataset_splitting(normal_data, attack_data,):
    """
    Splits data into train / validation / test with proper shuffling.

    Training:
        - 70% normal only
    Validation:
        - 15% normal + 85% attack
    Test:
        - 15% normal + 15% attack
    """
    # converting labels into tensors for compatability
    normalLabels = torch.zeros(len(normal_data), dtype=torch.long)
    attackLabels = torch.ones(len(attack_data), dtype=torch.long)


    # --------------------------------------------------
    # Shuffle NORMAL data
    # --------------------------------------------------
    normal_perm = torch.randperm(len(normal_data))
    normal_data = normal_data[normal_perm]
    normalLabels = normalLabels[normal_perm]

    # --------------------------------------------------
    # Shuffle ATTACK data
    # --------------------------------------------------
    attack_perm = torch.randperm(len(attack_data))
    attack_data = attack_data[attack_perm]
    attackLabels = attackLabels[attack_perm]

    # --------------------------------------------------
    # NORMAL splits
    # --------------------------------------------------
    num_normal = len(normal_data)
    train_size = int(0.7 * num_normal)
    val_size = int(0.15 * num_normal)

    train_data = normal_data[:train_size]
    trainingLabels = normalLabels[:train_size]

    val_normal_data = normal_data[train_size:train_size + val_size]
    val_normal_labels = normalLabels[train_size:train_size + val_size]

    test_normal_data = normal_data[train_size + val_size:]
    test_normal_labels = normalLabels[train_size + val_size:]

    # --------------------------------------------------
    # ATTACK splits
    # --------------------------------------------------
    num_attack = len(attack_data)
    val_attack_size = int(0.85 * num_attack)
    test_attack_size = int(0.15 * num_attack)

    val_attack_data = attack_data[:val_attack_size]
    val_attack_labels = attackLabels[:val_attack_size]

    test_attack_data = attack_data[val_attack_size:val_attack_size + test_attack_size]
    test_attack_labels = attackLabels[val_attack_size:val_attack_size + test_attack_size]

    # --------------------------------------------------
    # Combine validation and test
    # --------------------------------------------------
    val_data = torch.cat((val_normal_data, val_attack_data), dim=0)
    validationLabels = torch.cat((val_normal_labels, val_attack_labels), dim=0)

    test_data = torch.cat((test_normal_data, test_attack_data), dim=0)
    testLabels = torch.cat((test_normal_labels, test_attack_labels), dim=0)

    return (
        train_data,
        val_data,
        test_data,
        trainingLabels,
        validationLabels,
        testLabels,
    )



def phase1_data_prepration(n, use_stats):
    '''
    This function prepares the data for training and evaluation.
        Packets preprocessing:
            We should have a function that load the dataset in form of raw packets, and return it in shape of (m, n)
            And it should truncate/pad each packet to M bytes
            It is already implemented before, but we will just need to adapt it here.    
    '''
    def load_all_modules():
        # print('loading all necessary modules')
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        sheet1_codes_path = os.path.abspath(os.path.join(curr_dir,'..', '..', '..','Assignment1','Abdelaziz_Codes' ,'Sheet1_codes'))
        sys.path.append(sheet1_codes_path)
        
    
    def pad_or_truncate_packet(packet, n):
        '''
            This function pads or truncates a packet to have exactly n bytes.
            Input:
                - packet: byte array of the packet
            Output:
                - processed_packet: byte array of the packet with exactly n bytes
        '''
        if len(packet) > n:
            return packet[:n]
        elif len(packet) < n: 
            return packet + bytes(n - len(packet))
        else:
            return packet

        
    def extract_statistical_features(byte_sequence):
        """
        Extract 5 statistical features from a raw byte sequence (before trim/pad).
        """

        # -----------------------------
        # SAFE conversion to uint8 array
        # -----------------------------
        if isinstance(byte_sequence, (bytes, bytearray)):
            x = np.frombuffer(byte_sequence, dtype=np.uint8)
        else:
            x = np.asarray(byte_sequence, dtype=np.uint8)

        # -----------------------------
        # 1) Original byte length
        # -----------------------------
        original_length = len(x)

        if original_length == 0:
            return np.zeros(5, dtype=np.float32)

        # -----------------------------
        # 2) Most frequent bytes
        # -----------------------------
        byte_counts = Counter(x)
        counts = sorted(byte_counts.values(), reverse=True)

        most_freq_count = counts[0]
        second_most_freq_count = counts[1] if len(counts) > 1 else 0

        # -----------------------------
        # 3) Byte entropy
        # -----------------------------
        probs = np.array(list(byte_counts.values()), dtype=np.float32) / original_length
        entropy = -np.sum(probs * np.log2(probs + 1e-12))

        # -----------------------------
        # 4) Mean absolute byte difference
        # -----------------------------
        if original_length > 1:
            mean_abs_diff = np.mean(np.abs(np.diff(x)))
        else:
            mean_abs_diff = 0.0

        return np.array([
            original_length,
            most_freq_count,
            second_most_freq_count,
            entropy,
            mean_abs_diff
        ], dtype=np.float32)


    def load_and_label_data(n, use_stats):
        normal_pcap_path = "../../DataSets/2017QUT_S7comm/LabelledDataset/20161219132813_control_set"
        attacked_pcap_path  = "../../DataSets/2017QUT_S7comm/LabelledDataset/20161215163606_s7_process_attacks"
        load_all_modules()
        from utilities import generate_bytes_array_from_packet_list

        # check if .npy files already exist, read it if yes, else generate it from pcap
        print(f"[*] Loading normal packets...")
        if os.path.exists("all_packets_control.npy"):
            data_normal_arrays = np.load("all_packets_control.npy", allow_pickle=True)
        else:
            data_normal_arrays = generate_bytes_array_from_packet_list(normal_pcap_path, pad = False, label = 'control')
    

        print(f"[*] Loading attacked packets...")
        data_attacked = []
        if os.path.exists("all_packets_attack.npy"):
            data_attacked = np.load("all_packets_attack.npy", allow_pickle=True)
        else:
            data_attacked = generate_bytes_array_from_packet_list(attacked_pcap_path, pad = False, label = 'attack')
        


        labeled_data_normal =[ ]
        labeled_data_attack =[ ]
        features_normal = []
        features_attack = []
        for array in data_normal_arrays:
            for pkt in array:
                if use_stats:
                    stats = extract_statistical_features(pkt)
                    features_normal.append(stats)
                labeled_data_normal.append( (pad_or_truncate_packet(pkt, n), 'normal') ) 

        if TESTING:
            for array in data_attacked[-2]:# this logic should be the same as normal.
                # for pkt in array:
                if use_stats:
                    stats = extract_statistical_features(array)
                    features_attack.append(stats)
                labeled_data_attack.append( (pad_or_truncate_packet(pkt, n), 'attack') )
        else:
            for array in data_attacked:
                for pkt in array:
                    if use_stats:
                        stats = extract_statistical_features(pkt)
                        features_attack.append(stats)
                    labeled_data_attack.append( (pad_or_truncate_packet(pkt,n), 'attack') )
        return labeled_data_normal, labeled_data_attack, features_normal, features_attack

    load_all_modules()
    return load_and_label_data(n, use_stats)

def prepare_tensors(data, n, m):
    '''
        This function converts the data to tensors.
        Input:
            - data: list of tuples (packet, label)
            - n: number of bytes per packet
            - m: number of packets per sample
        Output:
            - data_tensor: tensor of shape (num_samples, 1, m, n)
    '''
    print(f'[*] Converting data to tensors...\n data shape is : {len(data)}')
    print(type(data))
    print(f'shape of first packet: {len(data[0][0])} , label: {data[0][1]}')
    num_samples = len(data) // m
    data_tensor = torch.zeros((num_samples, 1, m, n), dtype=torch.float32)
    for i in range(num_samples):
        for j in range(m):
            packet, _ = data[i * m + j]
            data_tensor[i, 0, j, :] = torch.tensor(
                np.frombuffer(packet, dtype=np.uint8),
                dtype=torch.float32
            ) / 255.0
    return data_tensor


