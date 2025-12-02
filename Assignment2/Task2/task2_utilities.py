import numpy as np
from dataclasses import dataclass
from enum import Enum
import random
from dataclasses import dataclass
from typing import List, Literal

# -----------------------------
# Field detection
# -----------------------------
FieldType = Literal["static", "dynamic"]

@dataclass
class UnitField:
    idx: int
    field_type: FieldType


@dataclass
class MergedField:
    start_idx: int
    end_idx: int
    field_type: FieldType


@dataclass
class KeywordCandidate:
    field_id: int
    start_idx: int
    end_idx: int
    field_type: FieldType

    # return len
    def __len__(self):
        return self.end_idx - self.start_idx + 1


class FieldType(Enum):
    STATIC = 0
    DYNAMIC = 1


# -----------------------------
# Task 2.a utility functions
# -----------------------------
# Build keyword candidates
# -----------------------------
def build_keyword_candidates(merged_fields):
    return [
        KeywordCandidate(
            field_id=i,
            start_idx=f.start_idx,
            end_idx=f.end_idx,
            field_type=f.field_type,
        )
        for i, f in enumerate(merged_fields)
    ]

def generate_dummy_aligned_messages(
    n_msgs: int = 100,
    msg_len: int = 20,
    static_ratio: float = 0.6,
    true_keyword_idx: int = 2,
    num_message_types: int = 3, # possible types for the message to simulate the real behavior. 
):
    """
    Generates dummy aligned messages with:
      - static fields (same value across all messages)
      - semi-static fields (same value but with noise)
      - dynamic fields (fully random)
      - ONE TRUE KEYWORD FIELD -> realistic clustering and pm
    """

    msgs = np.zeros((n_msgs, msg_len), dtype=np.uint8)

    # ----------------------------
    # 1. Assign static/dynamic positions
    # ----------------------------
    num_static = int(static_ratio * msg_len)
    static_positions = set(random.sample(range(msg_len), num_static))

    # Ensure true keyword field is not static
    if true_keyword_idx in static_positions:
        static_positions.remove(true_keyword_idx)

    dynamic_positions = set(range(msg_len)) - static_positions - {true_keyword_idx}

    # ----------------------------
    # 2. Fill STATIC fields
    # ----------------------------
    for col in static_positions:
        static_value = random.randint(0, 255)
        msgs[:, col] = static_value

    # ----------------------------
    # 3. Fill TRUE KEYWORD FIELD
    # ----------------------------
    possible_types = [random.randint(1, 255) for _ in range(num_message_types)]
    for i in range(n_msgs):
        msgs[i, true_keyword_idx] = random.choice(possible_types)

    # ----------------------------
    # 4. Fill DYNAMIC fields
    # ----------------------------
    for col in dynamic_positions:
        msgs[:, col] = np.random.randint(0, 255, size=n_msgs)

    return msgs


# -----------------------------
# Detect static/dynamic unit fields
# -----------------------------
def detect_unit_fields(aligned_msgs):
    unit_fields = []

    for col in range(aligned_msgs.shape[1]):
        col_vals = aligned_msgs[:, col]
        unique_vals = np.unique(col_vals)

        if len(unique_vals) == 1:
            ftype = "static"
        else:
            ftype = "dynamic"

        unit_fields.append(UnitField(idx=col, field_type=ftype))

    return unit_fields


# -----------------------------
# Merge static fields
# -----------------------------
def merge_static_fields(unit_fields: List[UnitField]) -> List[MergedField]:
    """
    Merge only consecutive static fields (Task 1.c).
    Dynamic fields remain single fields.
    """
    merged = []

    i = 0
    n = len(unit_fields)

    while i < n:
        uf = unit_fields[i]

        # If static → start merging run
        if uf.field_type == "static":
            start = uf.idx
            end = uf.idx

            # expand run only through static + consecutive columns
            j = i + 1
            while j < n and unit_fields[j].field_type == "static" and unit_fields[j].idx == end + 1:
                end = unit_fields[j].idx
                j += 1

            merged.append(MergedField(start_idx=start, end_idx=end, field_type="static"))
            i = j  # skip ahead

        else:
            # dynamic → one field by itself
            merged.append(MergedField(start_idx=uf.idx, end_idx=uf.idx, field_type="dynamic"))
            i += 1

    return merged

# -----------------------------

def generate_dummy_data_for_task2(
    n_msgs: int = 100,
    msg_len: int = 20,
    static_ratio: float = 0.6,
):
    aligned_msgs = generate_dummy_aligned_messages(
        n_msgs=n_msgs,
        msg_len=msg_len,
        static_ratio=static_ratio,
    )

    unit_fields = detect_unit_fields(aligned_msgs)
    merged_fields = merge_static_fields(unit_fields)
    keyword_candidates = build_keyword_candidates(merged_fields)

    return aligned_msgs, unit_fields, merged_fields, keyword_candidates

# -----------------------------
# Task2.b utility functions
# -----------------------------

def cluster_by_keyword(aligned_msgs, kc):
    """
    Logic:
        Clusters messages by the value of the keyword candidate field.
        i.e.: [ 01, 02, 03, 04], [01, 05, 02, 06], and kc = 0 (01)
        would produce clusters:
            { (01,): [0, 1] }
        So, messages with same value in the keyword field are grouped together.
    Args:
        aligned_msgs: np.ndarray of shape (n_msgs, msg_len)
        kc: KeywordCandidate object
    Returns: dict { field_value_tuple -> list_of_msg_indices }
    """
    clusters = {}
    start = kc.start_idx
    end   = kc.end_idx

    for i, msg in enumerate(aligned_msgs):
        # get the field value for this keyword candidate
        field_value = tuple(msg[start:end + 1])  # make hashable

        # if not exist before, create new cluster
        if field_value not in clusters:
            clusters[field_value] = []

        # add message index to the cluster
        clusters[field_value].append(i)
    return clusters

def generate_all_clusters(
    aligned_msgs,
    filtered_keywords: List[KeywordCandidate],
):
    '''
        Input: 
            aligned_msgs: np.ndarray of shape (n_msgs, msg_len)
            filtered_keywords: List of KeywordCandidate objects after filtering
        Logic: 
            For each keyword candidate, cluster the messages by the value of that keyword field.
        Output:
            final_clusters: List of dicts, each dict corresponds to a keyword candidate and maps
                            field_value_tuple -> list_of_msg_indices
    '''


    print("Performing clustering...\n")
    final_clusters = []
    for kc in filtered_keywords:
        clusters = cluster_by_keyword(aligned_msgs, kc)
        final_clusters.append(clusters)

    # clusters at index i are for keyword candidate at index i
    return final_clusters

def compute_similarity_matrix(aligned_msgs):
    """
    First we should compute the similarity matrix for the aligned messages.
        - For each pair of aligned messages, compare its similarity to each other, which can be computed as following: 
            - number of identical bytes / total number of bytes in both messages (here they have the same size always)
            - This will give us symmetric similarity matrix of size N x N, where N is the number of aligned messages.
    Computes full NxN similarity matrix for aligned messages.
    sim[i][j] = (# matching bytes) / length
    """
    msgs = aligned_msgs
    n, L = msgs.shape # n = number of messages, L = length of each message
    sim = np.zeros((n, n), dtype=float) # similarity matrix

    for i in range(n):
        # vectorized comparison with all messages
        matches = (msgs == msgs[i])  # boolean matrix

        # count matches per row
        identical_counts = matches.sum(axis=1) # sum along columns for each row ( This is a vector of size n)

        # compute similarity
        sim[i] = identical_counts / L
    
    # print('ensure that similarity matrix is correct')
    # print(sim)
    # print(msgs[0])
    # print(msgs[1])
    # print(msgs[2])
    # print(identical_counts)
    return sim


def compute_similarity_scores_for_keyword(sim_matrix, clusters, threshold=0.9):
    '''
        Input: 
            sim_matrix: similarity matrix of shape (n_msgs, n_msgs)
            clusters: dict of clusters for a specific keyword
                i.e. { field_value_tuple -> list_of_msg_indices }
            threshold: similarity threshold to determine matches
        Logic: 
            1. create a map, where each message index maps to its cluster id
            2. iterate over each pair of messages (i, j) in the similarity matrix
                - if both messages are in the same cluster, add their similarity score to inner_scores
                - else, add their similarity score to inter_scores
            3. compute FMR and FNMR based on the threshold
            4. compute pm = 1 - (FMR + FNMR) / 2
        Output:
            pm: similarity score for the keyword's clusters
    '''

    # performing 
    inner_scores = []
    inter_scores = []

    # Flatten cluster structure to: msg_index -> cluster_id
    # This allows easy lookup of which cluster a message belongs to.
    # i.e. [msg0 -> cluster0, msg1 -> cluster2, ...]
    msg_to_cluster = {}
    for c_id, (_, members) in enumerate(clusters.items()):
        for m in members:
            msg_to_cluster[m] = c_id

    # assign unclustered messages to a special cluster to avoid neglecting empty clusters and length mismatch
    for msg_index in range(sim_matrix.shape[0]):
        if msg_index not in msg_to_cluster:
            msg_to_cluster[msg_index] = -1   # “noise” cluster

    # number of messages
    n = sim_matrix.shape[0]


    # Compare each pair (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            # extracting the similarity score
            s = sim_matrix[i][j]
            if msg_to_cluster[i] == msg_to_cluster[j]:
                # if two messages are in the same cluster
                # add their score to inner scores
                inner_scores.append(s)
            else:
                # if two messages are in different clusters
                # add their score to inter scores
                inter_scores.append(s)

    # Avoid division by zero
    if len(inner_scores) == 0 or len(inter_scores) == 0:
        return 0

    # FMR = inter scores misclassified as match
    # the condition returns 1 if s > threshold for each element, and I count the sum of all of them
    # The meaning of FMR is that, they have high similarity score, but they are not in the same
    # cluster, so this mean false matching rate.
    FMR = sum(s > threshold for s in inter_scores) / len(inter_scores)

    # FNMR = inner scores misclassified as non-match
    # The meaning of FNMR is that, they have low similarity score, but they are in the same
    # cluster, so this mean false non-matching rate.
    FNMR = sum(s < threshold for s in inner_scores) / len(inner_scores)

    # final metric pm
    pm = 1 - ((FMR + FNMR) / 2.0)

    return pm



def compute_message_similarity_scores(sim_mat, final_clusters):
    '''
        Input:
            sim_mat: similarity matrix of shape (n_msgs, n_msgs)
            final_clusters: List of all clusters for all keywords
                i.e. final_clusters[i] are clusters correspond to filtered_keywords[i]
        Output:
            pms: List of pm scores for each keyword, same order as final_clusters
        Logic:
            For each keyword's clusters, compute the similarity score pm using the similarity matrix.

    '''
    pms = []
    # iterate over each keyword's clusters
    for cluster in final_clusters:
        pm = compute_similarity_scores_for_keyword(
            sim_mat,
            cluster,
        )
        pms.append(pm)
    return pms


def compute_remote_coupling(clusters_client, clusters_server, client_to_server_mapping):
    """
    Computes the remote coupling probability pr.
    3. compute PR scores for each keyword
        - main idea here is to check:
            - do the same type of requests tend to produce the same response ?
            - because this should be the case.
        - after clustering the client messages and server messages separately,
        - we should compute for for one cluster of size N:
            - for each message in the cluster:
                - there should be a response message
                - so we should count to which Cluster Cj it belongs in the server
                - and we should assign that cluster the server cluster with highest count (M)
            - then we should compute PR as following:
                - PR = M / N
    """
    # Reverse-lookup: server msg -> server cluster ID
    # same idea, I want to know for each message in the server, 
    # what cluster it belongs to.
    server_msg_to_cluster = {}
    for sc_id, (_, members) in enumerate(clusters_server.items()):
        for idx in members:
            server_msg_to_cluster[idx] = sc_id

    pr_values = []

    # For each client cluster
    for field_values, client_members_indicies in clusters_client.items():
        # each iteration we should get server messages corresponding to the client messages
        # i.e. client_members_indicies = [0, 1, 5] -> client messages at index 0, 1, and 5

        # for each request message in the client cluster,
        # find the corresponding response message in the server using the mapping
        server_indices = [
            client_to_server_mapping[clientIdx]
            for clientIdx in client_members_indicies
            if clientIdx in client_to_server_mapping
        ]

        # if no server messages correspond to this client cluster, skip
        if len(server_indices) == 0:
            continue

        # Count how many fall into each server cluster
        # keys: server cluster IDs
        # values: counts
        cluster_counts = {}
        for s in server_indices:
            if s in server_msg_to_cluster: # so it is mapped to a server cluster
                sc = server_msg_to_cluster[s] # extract the cluster
                cluster_counts[sc] = cluster_counts.get(sc, 0) + 1 # get current count, or 0 if not exist, then add 1.
        print(cluster_counts)

        # If no server clusters found, PR = 0 for this client cluster
        if len(cluster_counts) == 0:
            pr_values.append(0)
            continue

        # Best match (dominant server cluster)
        M = max(cluster_counts.values())
        print(M)
        N = len(client_members_indicies) # size of client cluster
        pr_cluster = M / N
        pr_values.append(pr_cluster)

    if len(pr_values) == 0:
        return 0.0
    
    # Final PR is average over all client clusters (this was not clear in the task, but I assumed it should be like this).
    return sum(pr_values) / len(pr_values)



def create_mapping(msgs_client, msgs_server):
    '''
        Input:
            msgs_client: List of client messages (list of lists)
            msgs_server: List of server messages (list of lists)
        Logic:
            map messages based on the 4th and 5th index values, they are realated from viewing the pcap files. 
        Output:
            mapping_indicies: dict mapping from client message index to server message index
    
    '''
    mapping_indicies = {} # client idx to server index
    print("Sample msgs_client:")
    print(f'len client msgs: {len(msgs_client)}')
    print(f'len server msgs: {len(msgs_server)}')
    min_len = min(len(msgs_client), len(msgs_server))
    for i in range (min_len):
        for j in range(4,5):
            if msgs_client[i][j] is not None and msgs_client[i][j] == msgs_server[i][j] and msgs_client[i][j+1] is not None and msgs_client[i][j+1] == msgs_server[i][j+1]:
                print(f"msg {i} client: {hex(msgs_client[i][j])} {hex(msgs_client[i][j+1])}")
                print(f"msg {i} server: {hex(msgs_server[i][j])} {hex(msgs_server[i][j+1])}")
                print("  --> Match")
                mapping_indicies[i] = i
            else:
                print("  --> No Match")
                continue
    
    return mapping_indicies

def compute_pr_for_keyword(msgs_client, msgs_server, clusters_client, clusters_server):

    # I must have mapping from client msg idx to server msg idx
    client_to_server_mapping = create_mapping(msgs_client, msgs_server)
    pr = compute_remote_coupling(clusters_client, clusters_server, client_to_server_mapping)
    print(f"Remote coupling probability pr: {pr}")
    return pr

def compute_prs(clusters_client, clusters_server, msgs_client, msgs_server):
    prs = []

    # get the minimum length (normally they should be the same, but to avoid index errors)
    min_len = min(len(clusters_client), len(clusters_server))
    for i in range(min_len):
        pr = compute_pr_for_keyword(
            msgs_client, msgs_server, clusters_client[i], clusters_server[i]   
        )
        prs.append(pr)
    return prs


def compute_structure_coherence(aligned_msgs, clusters):
    """
    ps = structure coherence score
    Measures structural similarity inside each cluster by checking
    how many positions differ from a reference message.
    """

    msg_len = aligned_msgs.shape[1]
    ps_values = []

    for field_values, members in clusters.items():

        if len(members) < 2:
            # A single message cluster is fully coherent structurally
            ps_values.append(1.0)
            continue

        ref = aligned_msgs[members[0]]  # extract first element as reference message
        gap_counts = []

        # think of it as matrix of size (num_members, msg_len)
        for msg_idx in members:
            diffs = 0
            msg = aligned_msgs[msg_idx]
            # diff += index of None in msg which is not None in ref or vice versa
            for i in range(msg_len):
                if (msg[i] is None) != (ref[i] is None):
                    diffs += 1
            gap_counts.append(diffs)
        avg_gap_count = sum(gap_counts) / len(members)
        len_aligned_msgs = len(ref)
        ps_cluster = 1 - (avg_gap_count / len_aligned_msgs)
        ps_values.append(ps_cluster)

    if len(ps_values) == 0:
        return 0.0

    # Final ps is average over all clusters, this was not specified in task, but I assumed it should be like this.
    return sum(ps_values) / len(ps_values)

def compute_ps_scores(aligned_msgs, clusters):
    '''
        Input: 
            aligned_msgs: np.ndarray of shape (n_msgs, msg_len)
            clusters: List of all clusters for all keywords
                i.e. clusters[i] are clusters correspond to filtered_keywords[i]
        Logic: 
            compute the structure coherence score ps for that keyword's clusters.
        Output:
            ps_list: List of ps scores for each keyword candidate
    '''
    ps_list = []
    for c in clusters:
        ps = compute_structure_coherence( aligned_msgs, c)
        ps_list.append(ps)
    return ps_list


def determine_empty_indices(aligned_msgs):
    """
    Determine indices which have None values in any message.
    """
    none_indices_client = set() # set to avoid duplicates
    for _, msg in enumerate(aligned_msgs):
        for j, field in enumerate(msg):
            if field is None:
                none_indices_client.add(j)
    # none_indices_server = set()
    # for i, msg in enumerate(msgs_server):
    #     for j, field in enumerate(msg):
    #         if field is None:
    #             none_indices_server.add(j)
    return none_indices_client


def print_physical_bytes_after_keyword(aligned_msgs: np.ndarray, keyword: KeywordCandidate):
    """
    Print the physical bytes in the aligned messages that lie after the end index of the given keyword.
    """
    print("Physical bytes after the keyword:")

    # print all bytes which are not none
    for i , msg in enumerate(aligned_msgs):
        print(f'physical data of message number {i}')
        for j in range(keyword.end_idx + 1, len(msg)):
            if msg[j] is not None:
                print(hex(msg[j]), end=' ')
        print()  # new line per message