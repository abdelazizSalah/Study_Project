'''
@Author: Abdelaziz Neamatallah
@Date: 01.12.25
@Desc: The goal of this script is to implement N-gram based Detection Method
- The main goal is to use a mixture of N-grams to model the bytes of the network packets and based on the 
generated model, detect whether a given packet is normal or anomalous.
- during training the method, we should observe each distinct n-gram seen in the sequence of bytes in each network packet 
in the training data, and record it in a space-efficien Bloom filter, also we should record the count 
of each seen n-gram during the traininig phase.
- Bloom filter is a data structure that is spaced-optimized version of hashing.
It is usually used to test whether an element is a member of a set.

- during testing phase, each packet should be scored by measuring how many n-grams were found in it.
- Score = (sum of (N-grams found in the packet * normalized weight of how often this n-gram was seen during training)) / Total number of n-grams in the current packet.


- network packet should be classified as anomalous if its score is below a certain threshold.
- training should search for the optimal threshold that maximizes the detection rate while minimizing the false positive rate.
- Finally, for the test-set, the script should determine for each packet whether it is normal or anomalous based on the learned model and the optimal threshold.
'''

'''
Logic analysis: 
1. I should split my work on 3 main phases:
    1.1 Training phase:
        - Extract all n-grams from each packet in the training set.
        - Add every distinct n-gram to the Bloom filter.
        - Count how many times each n-gram appears across all packets.
        - compute the weight (wi) for each n-gram based on its frequency.
            - wi = frequency of n-gram i / total number of n-grams
        - determine the optimal threshold for classification based on training data.
    1.2 Testing phase:
        - for every test packet: 
            - Extract its n-grams. 
            - Using bloom-filter check which of its n-grams were seen during training. 
            - For each n-gram:
                - If it was seen during training:
                    - get its weight (wi) from the training phase.
                    - compute score contribution: (N_seen_i * wi) # I assume N_seen_i = frequency of n-gram i in the test packet
            - Compute total score for the packet:
                - Score = (sum of (N-grams found in the packet * normalized weight of how often this n-gram was seen during training)) / Total number of n-grams in the current packet.
                - Score should be between [0, 1]
    1.3 Classification phase:
        - Classify each test packet as normal or anomalous based on the optimal threshold determined during training.
        - If Score < threshold: classify as anomalous else normal.
'''

'''
Finding optimal threshold:
- define a possible range for thresholds from 0 to 1 with small step size (e.g., 0.01).
- for each threshold in the defined range:
    - classify each packet in the training set based on the current threshold.
    - calculate detection rate (true positive rate) and false positive rate for the current threshold.
    - store the threshold that gives the best trade-off between detection rate and false positive rate.
'''
TESTING = True
import argparse
import numpy as np
from collections import defaultdict
from math import log2
from scapy.all import rdpcap
import os
import pickle

# ----------------------------------------------------------
# BLOOM FILTER
# ----------------------------------------------------------
class BloomFilter:
    def __init__(self, size=500000, hash_count=5):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = np.zeros(size, dtype=bool)

    def _hashes(self, ngram):
        h1 = hash(ngram)
        for i in range(self.hash_count):
            yield (h1 + i * 7919) % self.size

    def add(self, ngram):
        for h in self._hashes(ngram):
            self.bit_array[h] = True

    def __contains__(self, ngram):
        return all(self.bit_array[h] for h in self._hashes(ngram))


# ----------------------------------------------------------
# N-GRAM EXTRACTION
# ----------------------------------------------------------
def extract_ngrams(byte_seq, n):
    return [byte_seq[i:i+n] for i in range(len(byte_seq) - n + 1)]


# ----------------------------------------------------------
# TRAINING PHASE
# ----------------------------------------------------------
def train_ngram_models(train_packets, n):
    '''
      1.1 Training phase:
        - Extract all n-grams from each packet in the training set.
        - Add every distinct n-gram to the Bloom filter.
        - Count how many times each n-gram appears across all packets.
        - compute the weight (wi) for each n-gram based on its frequency.
            - wi = frequency of n-gram i / total number of n-grams
        - determine the optimal threshold for classification based on training data.
    '''


    bloom = BloomFilter()
    ngram_count = {}

    total_seen = 0 # total number of distinct n-grams seen during training

    for pkt in train_packets:
        for i in range(2,n+1):# generating many n-grams till the maximum n (include n).
            # pass only first element in the tuple (packet, label)
            grams = extract_ngrams(pkt[0], i)
            for g in grams:
                g = bytes(g)
                if g not in ngram_count:
                    bloom.add(g)
                    ngram_count[g] = 1
                    total_seen += 1
                else:
                    ngram_count[g] += 1

    # normalized weights
    weights = {curr_gram: c / total_seen for curr_gram, c in ngram_count.items()}

    return bloom, weights


# ----------------------------------------------------------
# SCORE A PACKET
# ----------------------------------------------------------
def score_packet(packet, bloom, weights, n):
    T = 0
    score_sum = 0
    for i in range(2,n+1):# generating many n-grams till the maximum n (include n).
        # # print(f'current packet is: {packet}')
        grams = extract_ngrams(packet, i)
        T += len(grams) # total number of n-grams in the current packet.
        if len(grams) == 0:
            continue


        local_count = {}
        # print(f'First 5 n-grams: {grams[:5]}, and its length is {len(grams)}')  # # print first 5 n-grams for debugging
        for g in grams:
            g = bytes(g)
            local_count[g] = local_count.get(g, 0) + 1

        for g, cnt in local_count.items():
            if g in bloom: # if it was seen during training, get N_seen and weight
                w = weights.get(g, 0)
                score_sum += w * cnt

    final_score = score_sum / T
    return final_score


# ----------------------------------------------------------
# SEARCH FOR BEST THRESHOLD
# ----------------------------------------------------------

def compute_metrics(y_true, y_pred):
    """
    y_true: list of {normal,attack}
    y_pred: list of {normal,attack}

    percision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    accuracy  = (TP + TN) / (TP + TN + FP + FN)

    tp = true positive
    tn = true negative
    fp = false positive
    fn = false negative

    return: accuracy, precision, recall
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'attack' and yp == 'attack')
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'normal' and yp == 'normal')
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'normal' and yp == 'attack')
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 'attack' and yp == 'normal')
    accuracy = (tp + tn) / (tp + tn + fp + fn) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def find_best_threshold(test_set, scores):
    """
    test_set: list of (packet, true_label)
              true_label = 1 -> NORMAL
              true_label = 0 -> ATTACK
    """

    best_acc = -1
    best_prec = -1
    best_rec = -1
    best_f1 = -1
    best_t = None

    # -------------------------------------------------
    # 1. Compute scores for all test packets ONCE only
    # -------------------------------------------------
    # scores = [score_packet(pkt, bloom, weights, n) for pkt, _ in test_set]
    y_true = [label for _, label in test_set]

    # -------------------------------------------------
    # 2. Sweep thresholds between 0 and 1
    # -------------------------------------------------
    # # print how many normal, and how many attack labels are in y_true
    num_normal = sum(1 for label in y_true if label == 'normal')
    num_attack = sum(1 for label in y_true if label == 'attack')
    print(f'Number of normal packets: {num_normal}, Number of attack packets: {num_attack}\n total: {len(y_true)}')


    highest_prec = -1
    highest_rec = -1
    highest_acc = -1
    highest_f1 = -1
    for t in np.linspace(0, 1, 1000):
        # print(f'Evaluating threshold: {t:.4f}')
        # 3. Predict using threshold t
        y_pred = ['normal' if s >= t else 'attack' for s in scores] 

        # 4. Compute metrics
        acc, prec, rec, f1_score = compute_metrics(y_true, y_pred)

        # 5. Compare with previous best
        if prec > highest_prec:
            highest_prec = prec
        if rec > highest_rec:
            highest_rec = rec
        if acc > highest_acc:
            highest_acc = acc
        if f1_score > highest_f1:
            highest_f1 = f1_score
        
        if f1_score > best_f1: # if I prioritize recall, all of them will be normal, and if I prioritize precision, all of them will be attack.
            best_acc = acc
            best_prec = prec
            best_rec = rec
            best_f1 = f1_score
            best_t = t
    print(f'Highest Precision observed: {highest_prec}'
          f', Highest Recall observed: {highest_rec}'
        f', Highest Accuracy observed: {highest_acc}'
        f', Highest F1 Score observed: {highest_f1}')
    return best_t, best_acc, best_prec, best_rec, best_f1


# ----------------------------------------------------------
# TESTING
# ----------------------------------------------------------
def test_model(test_packets, bloom, weights, n, threshold):
    results = []
    for i, pkt in enumerate(test_packets):
        s = score_packet(pkt[0], bloom, weights, n)
        label = "NORMAL" if s >= threshold else "ATTACK"
        results.append((i, s, label))
    return results




def load_all_modules():
    import sys
    print(
                '''
        #############################################
        #-------------------------------------------#
        #       WELCOME TO AASP TOOL BOX  2025      #
        #   Electra & QUT S7comm Datasets will be   #
        #         easy for you to handle ;)         #
        #-------------------------------------------#
        #############################################

        '''
    )

    # print('loading all necessary modules')
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    sheet1_codes_path = os.path.abspath(os.path.join(curr_dir, '..', '..','Assignment1','Abdelaziz_Codes' ,'Sheet1_codes'))
    sys.path.append(sheet1_codes_path)
    

def load_and_label_data():
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
    
    labeled_data =[ ]
    i = 0
    for array in data_normal_arrays:
        for pkt in array:
            labeled_data.append( (pkt, 'normal') )
    i = 0 
    for array in data_attacked[-2]:
        # for pkt in array:
        labeled_data.append( (pkt, 'attack') )

    return labeled_data

# ----------------------------------------------------------
# MAIN ENTRY (CLI)
# ----------------------------------------------------------
def sheet3_task1():

    n = 3  # default n-gram size
    print('[*] Loading and labeling data...')
    labeled_data = load_and_label_data()
    
    # split data into training and testing sets
    # 80% should be from random indicies for training, and 20% for testing but I should not include same packets in both sets.
    # np.random.shuffle(labeled_data) # to ensure now random distribution and not first packets are normal and last are attack
    split_idx = int(0.8 * len(labeled_data))
    print(f'current split index is: {split_idx}, total data length is: {len(labeled_data)}')
    train_packets = labeled_data[:split_idx]
    test_packets = labeled_data[split_idx:] 
    if TESTING:
        # include 8000 from normal packets and 2000 from attack packets in the train set
        train_packets = [pkt for pkt in train_packets if pkt[1] == 'normal'][:8000] + [pkt for pkt in train_packets if pkt[1] == 'attack'][:2000]

        # include 2000 normal packets and 1000 attack packets in the test set
        test_packets = [pkt for pkt in train_packets if pkt[1] == 'normal'][:2000] + [pkt for pkt in test_packets if pkt[1] == 'attack'][:1000]
        # print(f'train packets {train_packets[:2]}')
        # print(f'test packets {test_packets[:2]}')
        print(f'len(test_packets): {len(test_packets)}, len(train_packets): {len(train_packets)}')


    print("[*] Training model...")
    if os.path.exists('bloom_filter.pkl') and os.path.exists('ngram_weights.pkl'):
        
        with open('bloom_filter.pkl', 'rb') as f:
            bloom = pickle.load(f)
        with open('ngram_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        print(f"[*] Loaded pre-trained model from disk. Number of n-grams in model: {len(weights)}")
        print('-------------------------------------')
    else:
        bloom, weights = train_ngram_models(train_packets, n)
        # save the bloom filter and weights to disk for future use
        # with open('bloom_filter.pkl', 'wb') as f:
        #     pickle.dump(bloom, f)
        # with open('ngram_weights.pkl', 'wb') as f:
        #     pickle.dump(weights, f)
        print(f"[*] Training completed. Number of n-grams in model: {len(weights)}")
        print('-------------------------------------')


    print("[*] Scoring test packets for threshold search...")
    test_scores = [score_packet(pkt[0], bloom, weights, n)
                    for pkt in test_packets]
    print("[*] Sample test scores:", test_scores[:10])
    print('-------------------------------------')

    # # Find optimal threshold
    best_t, best_acc, best_prec, best_rec, best_f1 = find_best_threshold(test_packets,scores=test_scores)
    print(f"[+] Optimal threshold found: {best_t:.4f} with Accuracy={best_acc:.4f}, Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1 Score={best_f1:.4f}")
    print('-------------------------------------')


    print("[*] Testing model...")
    results = test_model(test_packets, bloom, weights, n, best_t)

    # print how many normal and how many attack packets were detected
    normal_detected = sum(1 for _, _, label in results if label == 'NORMAL')
    attack_detected = sum(1 for _, _, label in results if label == 'ATTACK')
    print(f'Number of NORMAL packets detected: {normal_detected}, Number of ATTACK packets detected: {attack_detected}\n total: {len(results)}')
    print("[*] Sample results (index, score, label):", results[:10])

if __name__ == "__main__":
    sheet3_task1()
