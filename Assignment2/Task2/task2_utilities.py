
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
    Clusters messages by the value of the keyword candidate field.
    Returns: dict { field_value_tuple -> list_of_msg_indices }
    """
    clusters = {}
    start = kc.start_idx
    end   = kc.end_idx

    for i, msg in enumerate(aligned_msgs):
        field_value = tuple(msg[start:end + 1])  # make hashable

        if field_value not in clusters:
            clusters[field_value] = []

        clusters[field_value].append(i)

    return clusters

def clustering_messages_by_keywords(
    aligned_msgs,
    filtered_keywords: List[KeywordCandidate],
):
    
    print("Performing clustering...\n")
    final_clusters = []
    for kc in filtered_keywords:
        clusters = cluster_by_keyword(aligned_msgs, kc)
        final_clusters.append(clusters)

    # sort them by # of clusters ascending
    # the less clusters, the better the keyword
    final_clusters = [dict(sorted(c.items(), key=lambda item: len(item[1]))) for c in final_clusters]


    return final_clusters

def compute_similarity_matrix(aligned_msgs):
    """
    Computes full NxN similarity matrix for aligned messages.
    sim[i][j] = (# matching bytes) / length
    """
    msgs = aligned_msgs
    n, L = msgs.shape
    sim = np.zeros((n, n), dtype=float)

    for i in range(n):
        # vectorized comparison with all messages
        matches = (msgs == msgs[i])  # boolean matrix
        # count matches per row
        identical_counts = matches.sum(axis=1)
        sim[i] = identical_counts / L

    return sim


def compute_similarity_scores_for_keyword(sim_matrix, clusters, threshold=0.7):
    # performing 
    inner_scores = []
    inter_scores = []

    # Flatten cluster structure to: msg_index -> cluster_id
    msg_to_cluster = {}
    for c_id, (_, members) in enumerate(clusters.items()):
        for m in members:
            msg_to_cluster[m] = c_id

    
    # assign unclustered messages to a special cluster to avoid neglecting empty clusters and length mismatch
    for msg_index in range(sim_matrix.shape[0]):
        if msg_index not in msg_to_cluster:
            msg_to_cluster[msg_index] = -1   # “noise” cluster

    n = sim_matrix.shape[0]


    # Compare each pair (i < j)
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i][j]
            if msg_to_cluster[i] == msg_to_cluster[j]:
                inner_scores.append(s)
            else:
                inter_scores.append(s)

    # Avoid division by zero
    if len(inner_scores) == 0 or len(inter_scores) == 0:
        return 0

    # FMR = inter scores misclassified as match
    FMR = sum(s > threshold for s in inter_scores) / len(inter_scores)

    # FNMR = inner scores misclassified as non-match
    FNMR = sum(s < threshold for s in inner_scores) / len(inner_scores)

    # final metric pm
    pm = 1 - ((FMR + FNMR) / 2.0)

    return pm



def compute_message_similarity_scores(sim_mat, final_clusters):
    pms = []
    for cluster in final_clusters:
        pm = compute_similarity_scores_for_keyword(
            sim_mat,
            cluster,
        )
        pms.append(pm)
    return pms


def compute_remote_coupling(clusters_client, clusters_server, client_to_server):
    """
    Computes the remote coupling probability pr.
    """
    # Reverse-lookup: server msg -> server cluster ID
    server_msg_to_cluster = {}
    for sc_id, (_, members) in enumerate(clusters_server.items()):
        for idx in members:
            server_msg_to_cluster[idx] = sc_id

    pr_values = []

    # For each client cluster
    for _, client_members in clusters_client.items():
        # Find all corresponding server messages
        server_indices = [
            client_to_server[c]
            for c in client_members
            if c in client_to_server
        ]

        if len(server_indices) == 0:
            continue

        # Count how many fall into each server cluster
        cluster_counts = {}
        for s in server_indices:
            if s in server_msg_to_cluster:
                sc = server_msg_to_cluster[s]
                cluster_counts[sc] = cluster_counts.get(sc, 0) + 1

        if len(cluster_counts) == 0:
            pr_values.append(0)
            continue

        # Best match (dominant server cluster)
        dominant = max(cluster_counts.values())
        pr_cluster = dominant / len(server_indices)
        pr_values.append(pr_cluster)

    if len(pr_values) == 0:
        return 0.0

    return sum(pr_values) / len(pr_values)

def compute_pr_for_keyword(msgs_client, msgs_server, clusters_client, clusters_server):
    client_to_server = {i: i for i in range(min(len(msgs_client), len(msgs_server)))}
    pr = compute_remote_coupling(clusters_client, clusters_server, client_to_server)
    print(f"Remote coupling probability pr: {pr}")
    return pr

def compute_prs(clusters_client, clusters_server, msgs_client, msgs_server):
    prs = []

    # get the minimum length
    min_len = min(len(clusters_client), len(clusters_server))
    for i in range(min_len):
        pr = compute_pr_for_keyword(
            msgs_client, msgs_server, clusters_client[i], clusters_server[i]   
        )
        prs.append(pr)
    return prs


def compute_structure_coherence(kc, aligned_msgs, clusters):
    """
    ps = structure coherence score
    Measures structural similarity inside each cluster by checking
    how many positions differ from a reference message.
    """

    msg_len = aligned_msgs.shape[1]
    ps_values = []

    for _, members in clusters.items():

        if len(members) < 2:
            # A single message cluster is fully coherent structurally
            ps_values.append(1.0)
            continue

        ref = aligned_msgs[members[0]]  # reference message
        gap_count = 0
        total_positions = len(members) * msg_len

        for msg_idx in members:
            msg = aligned_msgs[msg_idx]
            # a "gap" is a structural difference
            diffs = np.sum(msg != ref)
            gap_count += diffs

        ps_cluster = 1 - (gap_count / total_positions)
        ps_values.append(ps_cluster)

    if len(ps_values) == 0:
        return 0.0

    return sum(ps_values) / len(ps_values)

def compute_ps_scores(filtered_keywords: List[KeywordCandidate], aligned_msgs):
    ps_list = []
    for kc in filtered_keywords:
        clusters = cluster_by_keyword(aligned_msgs, kc)
        ps = compute_structure_coherence(kc, aligned_msgs, clusters)
        ps_list.append(ps)
    return ps_list


def determine_empty_indices(aligned_msgs):
    """
    Determine indices which have None values in any message.
    """
    none_indices_client = set()
    for i, msg in enumerate(aligned_msgs):
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
    for _ , msg in enumerate(aligned_msgs):
        for j in range(keyword.end_idx + 1, len(msg)):
            if msg[j] is not None:
                print(msg[j], end=' ')
        print()  # new line per message