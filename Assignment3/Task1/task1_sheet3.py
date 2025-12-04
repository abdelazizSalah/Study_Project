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
TESTING = False
import sys
import numpy as np
import math
import os
import pickle
import hashlib
import mmh3  # MurmurHash3 library for fast hashing

# ----------------------------------------------------------
# BLOOM FILTER
# ----------------------------------------------------------
class BloomFilter:
    '''
        Bloom Filter implementation
        - uses multiple hash functions to set bits in a bit array
        - automatically computes the optimal size (m bits)
        - automatically computes the optimal number of hash functions (k)
        - stores bits efficiently using bytearray
        - uses double hashing (h1 + i * h2) to generate multiple hash values
        - supporting adding elements and checking membership
    '''
    def __init__(self, false_positive_rate=0.01, number_of_ngrams=10000):
        '''
            Constructor to initialize the Bloom filter
            false_positive_rate: desired false positive rate (between 0 and 1)
            number_of_ngrams: expected number of n-grams to be stored in the filter
        
        '''

        if number_of_ngrams <= 0 or false_positive_rate <=0 or false_positive_rate >=1:
            raise ValueError("Number of n-grams must be positive and false_positive_rate must be in (0,1) range")
        
        ln_p = math.log(false_positive_rate) # math.log is natural logarithm (ln)
        self.required_number_of_bits = int(-(number_of_ngrams * ln_p) / (math.log(2) ** 2)) # optimal size in bits The Bloom filter size formula is \(m=-(n*\ln (p))/(\ln (2)^{2})\)
        self.number_of_hash_functions = math.ceil((self.required_number_of_bits / number_of_ngrams) * math.log(2)) # k = (m / n) * ln(2)
        self.byte_array = bytearray(self.required_number_of_bits // 8 + 1)  # +1 to handle any remainder bits
        # bytes array is an efficient way to store bits in python, each byte has 8 bits, so number of bytes = ceil(number of bits / 8)
        # I could have used also boolean array, but 1 boolean = 1 byte, so it is less efficient in terms of space.


    def _set_certain_bit(self, bit_index):
        '''
            Set a certain bit in the Bloom filter
            bit_index: index of the bit to set
        
        '''

        byte_index = bit_index // 8
        bit_position = bit_index % 8
        self.byte_array[byte_index] |= (1 << bit_position) # set the bit using orwise operation by shifting 1 to the left by bit_position
    
    def _get_certain_bit(self, bit_index):
        '''
            Get the value of a certain bit in the Bloom filter
            bit_index: index of the bit to get
            returns: 1 if the bit is set, 0 otherwise
        '''
        byte_index = bit_index // 8
        bit_position = bit_index % 8
        return (self.byte_array[byte_index] >> bit_position) & 1 # get the bit using andwise operation by shifting right by bit_position till the 0th position which include my 1.

        
    def _hashes(self, data: bytes):
        '''
            Generate k hash values for the given data using double hashing
            data: byte sequence to be hashed
            yields: k hash values in the range [0, m-1]
        '''
        # First hash: MurmurHash3 (extremely fast, 64-bit output)
        h1, _ = mmh3.hash64(data, seed=0, signed=False)

        # Second hash: BLAKE2b (strong entropy, 64-bit output)
        blake = hashlib.blake2b(data, digest_size=8).digest()
        h2 = int.from_bytes(blake, "big") | 1  # ensure h2 is odd, to avoid cycles, and try to have all k hash values different

        # Produce k hash values using double hashing
        for i in range(self.number_of_hash_functions):
            yield (h1 + i * h2) % self.required_number_of_bits # return one value now, but remember where we left off to continue later.


    def add(self, ngram):
        '''
            Add an n-gram to the Bloom filter
            ngram: byte sequence representing the n-gram
        '''
        # I wanted to check if the element already exist, but I thought that due to collisions, it is better to just set the bits again.
        for h in self._hashes(ngram):
            self._set_certain_bit(h)

    def __contains__(self, ngram):
        '''
            Check if an n-gram is in the Bloom filter
            ngram: byte sequence representing the n-gram
            returns: True if n-gram is probably in the filter, False if definitely not in the filter
        '''
        return all(self._get_certain_bit(h) for h in self._hashes(ngram)) # all check that all bits are set to 1, which means the n-gram is probably in the filter.
        # I said probably because of false positives, and collisions. 

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

    ngram_count = {}
    total_seen = 0 # total number of distinct n-grams seen during training

    print(f'starting to train n-gram model...\n number of training packets: {len(train_packets)}')
    for pkt in train_packets:
        # for i in range(2,n+1):# generating many n-grams till the maximum n (include n).
            # pass only first element in the tuple (packet, label)
        grams = extract_ngrams(pkt[0], n)
        for g in grams:
            g = bytes(g)
            ngram_count[g] = ngram_count.get(g, 0) + 1
            total_seen += 1
    
    # add n-grams to bloom filter
    print('starting to add n-grams to bloom filter...')
    bloom = BloomFilter(false_positive_rate=0.01, number_of_ngrams=len(train_packets)*1000) # assuming each packet has 1000 distinct n-grams on average
    for curr_gram in ngram_count.keys():
        bloom.add(curr_gram)

    # normalized weights
    print('starting to normalize weights...')
    weights = {curr_gram: c / total_seen for curr_gram, c in ngram_count.items()} # I count the weights only to certan n-grams seen in training data
    # i.e. 0100 was seen 5 times, and total 2-grams are 100, then weight of 0100 = 5/100 = 0.05
    # then I will do 010011 for example which is 3-gram, so I will now count total seen different from the 100 of 2-grams, I will check how many 3-grams only.
            
    # sample from weights
    print(f'samples of weights are: {list(weights.items())[:20]}')

    return bloom, weights, ngram_count


# ----------------------------------------------------------
# SCORE A PACKET
# ----------------------------------------------------------
def score_packet(packet, bloom, weights, n): 
    '''
    Input:
        - packet: byte sequence of test packet
        - bloom: trained bloom filter from training phase
        - weights: trained n-gram weights from training phase
        - ngram_count: trained n-gram counts from training phase
        - n: maximum n-gram size
    Logic:
        - for every test packet:
            - Extract its n-grams. 
            - T = total number of n-grams in the current packet.
            - Using bloom-filter check which of its n-grams were seen during training. 
            - For each n-gram:
                - If it was seen during training:
                    - get its weight (wi) from the training phase.
                    - get N_seen_i from ngram_count which is frequency of n-gram i in the training phase.
                    - compute score contribution: (N_seen_i * wi) # I assume N_seen_i = frequency of n-gram i in the training packets
            - Compute total score for the packet:
                - Score = (sum of (N-grams found in the packet * normalized weight of how often this n-gram was seen during training)) / Total number of n-grams in the current packet.
                - Score should be between [0, 1]

        - returns: score of the packet based on the trained model
    
    '''
    
    T = 0 # represent total number of n-grams in the current packet.
    score_sum = 0
    # for i in range(2,n+1):# generating many n-grams till the maximum n (include n).
        # # print(f'current packet is: {packet}')
    grams = extract_ngrams(packet, n)
    T += len(grams) # total number of n-grams in the current packet.
    if len(grams) == 0:
        return 0

    
    for g in grams: # iterate for each gram in the current packet
        g = bytes(g)
        if g in bloom: # if it was seen during training
            w = weights.get(g, 0) # get its weight from the training phase
            # N_seen_i = ngram_count.get(g, 0) # get its count from the training phase
            score_sum += w  # compute score contribution: (N_seen_i * wi)
            # when I include N_seen_i, the score becomes > 1, and I think this is because we double count the counts.

    final_score = score_sum / T if T > 0 else 0
    return final_score


# ----------------------------------------------------------
# SEARCH FOR BEST THRESHOLD
# ----------------------------------------------------------

def compute_metrics(y_true, y_pred): # recheck this logic.
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
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def find_best_threshold(test_set, scores): # recheck this logic.
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
        label = "normal" if s >= threshold else "attack"
        results.append((i, s, label))
    # compute metrics for the test set
    y_true = [label for _, label in test_packets]
    y_pred = [label for _, _, label in results]
    acc, prec, rec, f1_score = compute_metrics(y_true, y_pred)
    print(f"[*] Test Set Results: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1 Score={f1_score:.4f}")
    # print how many normal and how many attack packets were detected
    normal_detected = sum(1 for _, _, label in results if label == 'normal')
    attack_detected = sum(1 for _, _, label in results if label == 'attack')
    print(f'Number of NORMAL packets detected: {normal_detected}, Number of ATTACK packets detected: {attack_detected}\n total: {len(results)}')
    print("[*] Sample results (index, score, label):", results[:10])

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
    
    labeled_data_normal =[ ]
    labeled_data_attack =[ ]
    for array in data_normal_arrays:
        for pkt in array:
            labeled_data_normal.append( (pkt, 'normal') ) 

    if TESTING:
        for array in data_attacked[-2]:# this logic should be the same as normal.
            # for pkt in array:
            labeled_data_attack.append( (pkt, 'attack') )
    else:
        for array in data_attacked:
            for pkt in array:
                labeled_data_attack.append( (pkt, 'attack') )
    return labeled_data_normal, labeled_data_attack

def compute_bloom_weights_and_counts(train_packets, n):
    
    print("[*] Training model...")
    if os.path.exists('bloom_filter.pkl') and os.path.exists('ngram_weights.pkl') and os.path.exists('ngram_count_training.pkl'):
        
        with open('bloom_filter.pkl', 'rb') as f:
            bloom = pickle.load(f)
        with open('ngram_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        with open('ngram_count_training.pkl', 'rb') as f:
            ngram_count_training = pickle.load(f)
        print(f"[*] Loaded pre-trained model from disk. Number of n-grams in model: {len(weights)}")
        print('-------------------------------------')
    else:
        bloom, weights, ngram_count_training = train_ngram_models(train_packets, n)
        # save the bloom filter and weights to disk for future use
        with open('bloom_filter.pkl', 'wb') as f:
            pickle.dump(bloom, f)
        with open('ngram_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
        # store ngram_count_training for future use
        with open('ngram_count_training.pkl', 'wb') as f:
            pickle.dump(ngram_count_training, f)
        print(f"[*] Training completed. Number of n-grams in model: {len(weights)}")
        print('-------------------------------------')
    return bloom, weights


def split_data(labeled_data_normal, labeled_data_attack, train_ratio=0.6, validation_ratio=0.2):
    '''
    Splits the labeled data into training, validation, and testing sets.
    Normal packets are split into (train ratio)% training, (validation ratio)% validation, and (1 - train ratio - validation ratio)% testing sets.
    Attack packets are split into 0% training, (validation ratio)% validation, and (1 - validation ratio)% testing sets.
    Returns:
        - train_packets: list of (packet, label) tuples for training
        - test_packets: list of (packet, label) tuples for testing
        - validation_packets: list of (packet, label) tuples for validation
    
    
    '''
    # split data into training, validaiton and testing sets
    split_idx_normal = int(train_ratio * len(labeled_data_normal)) # [ 60% normal packets for training, 20% for validation, 20% for testing ]
    train_normal, remaining_normal = labeled_data_normal[:split_idx_normal], labeled_data_normal[split_idx_normal:]
    split_idx_normal_val = int(validation_ratio * len(remaining_normal))
    validation_normal, test_normal = remaining_normal[:split_idx_normal_val], remaining_normal[split_idx_normal_val:]

    # split attack data into validation and testing sets only => [ 0% attack packets for training, 80% for validation, 20% for testing ]
    split_idx_attack = int(validation_ratio * len(labeled_data_attack))
    test_attack,validation_attack =  labeled_data_attack[:split_idx_attack], labeled_data_attack[split_idx_attack:]
   
    # assign the sets
    train_packets = train_normal  # only normal packets for training
    validation_packets = validation_normal + validation_attack # both normal and attack packets for validation
    test_packets = test_normal + test_attack # both normal and attack packets for testing
    if TESTING:
        # include 8000 from normal packets and 2000 from attack packets in the train set
        train_packets = train_packets[:80000]
        validation_packets = validation_normal[:20000] + validation_attack[:80000]
        # include 2000 normal packets and 2000 attack packets in the test set
        test_packets = test_normal[:10000] + test_attack[:30000]
        print(f'len(train_packets): {len(train_packets)}')
        print(f'len(validation_packets): {len(validation_packets)}')
        print(f'len(test_packets): {len(test_packets)}')
    return train_packets, test_packets, validation_packets
    
# ----------------------------------------------------------
# MAIN ENTRY (CLI)
# ----------------------------------------------------------
def sheet3_task1():
    # read n from command line arguments
    print('-------------------------------------')
    print("[*] Starting N-gram based Detection Method...")
    
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        # print a hint how to call it in command line
        raise ValueError("Please provide n-gram size as a command line argument. Example: python task1_sheet3.py 3")

        # n = 3  # default n-gram size
    # load and label data
    print('[*] Loading and labeling data...')
    labeled_data_normal, labeled_data_attack = load_and_label_data() 
    
    # split data into training and testing sets
    print("[*] Splitting data into training and testing sets...")
    train_packets, test_packets, validation_packets = split_data(labeled_data_normal, labeled_data_attack, train_ratio=0.6, validation_ratio=0.2)


    # compute bloom filter, weights, and ngram counts from training data
    print("[*] Train n-gram model and compute bloom filter, weights, and ngram counts...")
    bloom, weights = compute_bloom_weights_and_counts(train_packets, n)


    print("[*] Scoring validation packets for threshold search...")
    # validation should be on both normal and attack packets
    # normal packets should have high scores, while attack packets should have low scores
    validation_scores = [score_packet(pkt[0], bloom, weights, n)
                    for pkt in validation_packets]
    # save validation scores to a file
    with open('validation_scores.txt', 'w') as f:
        f.write("Index\tScore\tLabel\n")
        for index, (_, label) in enumerate(validation_packets):
            f.write(f"{index}\t{validation_scores[index]:.6f}\t{label}\n")
    print("[*] Sample validation scores:", validation_scores[:10])
    print('-------------------------------------')

    print("[*] Finding optimal threshold...")
    best_t, best_acc, best_prec, best_rec, best_f1 = find_best_threshold(validation_packets,scores=validation_scores)
    # save best threshold to a file
    with open('best_metrics.txt', 'w') as f:
        f.write(f"Best Threshold: {best_t:.6f}\n")
        f.write(f"Accuracy: {best_acc:.6f}\n")
        f.write(f"Precision: {best_prec:.6f}\n")
        f.write(f"Recall: {best_rec:.6f}\n")
        f.write(f"F1 Score: {best_f1:.6f}\n")
    print(f"[+] Optimal threshold on validation found: {best_t:.4f} with Accuracy={best_acc:.4f}, Precision={best_prec:.4f}, Recall={best_rec:.4f}, F1 Score={best_f1:.4f}")
    print('-------------------------------------')


    print("[*] Testing model with the best threshold from validation found...")
    results = test_model(test_packets, bloom, weights, n, best_t) 

    # save results to a file
    with open('test_results.txt', 'w') as f:
        f.write("Index\tScore\tLabel\n")
        for index, score, label in results:
            f.write(f"{index}\t{score:.6f}\t{label}\n")
    print("[*] Test results saved to test_results.txt")


   
if __name__ == "__main__":
    sheet3_task1()
