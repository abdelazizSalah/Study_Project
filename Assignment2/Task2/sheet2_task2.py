'''
@Author: Abdelaziz Neamatallah
@Date: 15.11.25
@ Description: Implementation of Task 2 of Sheet 2, which continue the operation of reverse engineering s7comm
    1. Exclude long keyword candidates
    2. Compute PM, PR, PS scores for each candidate
    3. Combine the scores to get final score for each candidate
    4. Select the candidate with highest final score as the most probable keyword
    5. Print the physical bytes after the best keyword candidate
'''


TESTING = True
import argparse
import numpy as np
from typing import List

from task2_utilities import (
    KeywordCandidate,
    generate_dummy_data_for_task2,
    compute_similarity_matrix,
    generate_all_clusters,
    compute_message_similarity_scores,
    compute_prs,
    compute_ps_scores,
    FieldType,
    determine_empty_indices,
    print_physical_bytes_after_keyword,
)


# -----------------------------
# Tasks
# -----------------------------
def task2_a_exclude_keywords_candidates(
    keyword_candidates: List[KeywordCandidate],
    none_indices: set,
    max_length: int = 4,
) -> List[KeywordCandidate]:
    """
    Since not every keyword candidate is legitimate, 
    So we should exclude candidates which are empty -- Contain None values in any message at their range
    Exclude keyword candidates longer than max_length.

    Input:
        keyword_candidates: List of KeywordCandidate objects
        none_indices: set of indices which contain None values in any message   
        max_length: maximum length of keyword candidates to keep
    Output:
        filtered_candidates: List of KeywordCandidate objects after filtering
    """
    print(f"Original candidates count: {len(keyword_candidates)}, and max_length: {max_length}")
    print(f"None indices: {none_indices}")
    filtered_candidates = [
        # also exclude those candidates which have None values in their range
        kc for kc in keyword_candidates # +1, if only 1 element then si = 2 ei = 2, then len = 2-2 + 1 = 1
        if len(kc) <= max_length and not any(idx in none_indices for idx in range(kc.start_idx, kc.end_idx + 1)) # + 1 because range in exclusive, and we should also include last index
    ]
    print(f"Filtered candidates count: {len(filtered_candidates)}")
    for idx, candidate in enumerate(filtered_candidates):
        print(f"Candidate {idx}: start_idx={candidate.start_idx}, end_idx={candidate.end_idx}, length={len(candidate)}")
    return filtered_candidates




def task2_b(filtered_keywords_client: List[KeywordCandidate], filtered_keywords_server: List[KeywordCandidate], aligned_msgs_client, aligned_msgs_server=None):
    '''
       First we should compute the similarity matrix for the aligned messages.
            - For each pair of aligned messages, compare its similarity to each other, which can be computed as following: 
                - number of identical bytes / total number of bytes in both messages (here they have the same size always)
                - This will give us symmetric similarity matrix of size N x N, where N is the number of aligned messages.
       Then we should cluster the messages based on the filtered keywords.
          i.e. one clustering process for each keyword
       Then for each clustering result:
            1. divide the similarity scores into intra-cluster and inter-cluster scores
               - intra-cluster scores: similarity scores between messages in the same cluster
               - inter-cluster scores: similarity scores between messages in different clusters
            2. compute PM scores for each keyword
               - using specific threshold t, we should compute FMR and FNMR
               - FMR = number of inter-cluster scores > t / total number of inter-cluster scores
               - FNMR = number of intra-cluster scores < t / total number of intra-cluster scores
               - PM = 1 - (FMR + FNMR) / 2
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
            4. compute PS scores for each keyword 
                - main idea here is that messages of same type should have same structure
                - so we should align again the messages in the same cluster
                - then count the average number of gaps in a message (total number of gaps / number of messages)
                - then we should compute PS as following:
                        - PS = 1 - (average number of gaps / total number of bytes in message)
       Finally we should return the PM, PR, PS scores for each keyword.
    
    '''

    # 1. compute similarity matrix
    sim_mat = compute_similarity_matrix(aligned_msgs_client)

    # 2. cluster messages based on keywords
    final_clusters_client = generate_all_clusters(
        aligned_msgs=aligned_msgs_client,
        filtered_keywords=filtered_keywords_client,
    )
    final_clusters_server = generate_all_clusters(
        aligned_msgs=aligned_msgs_server,
        filtered_keywords=filtered_keywords_server,
    )

    # 3. compute PM, PR, PS scores for each keyword
    pms_client = compute_message_similarity_scores(sim_mat, final_clusters_client)
    print("Final PM scores for each keyword client:\n", pms_client )

    pms_server = compute_message_similarity_scores(sim_mat, final_clusters_server)
    print("Final PM scores for each keyword server:\n", pms_server )

    prs = compute_prs(final_clusters_client, final_clusters_server, aligned_msgs_client, aligned_msgs_server)
    print("Final PR scores for each keyword:\n", prs )

    ps_client_list = compute_ps_scores( aligned_msgs_client, final_clusters_client)
    ps_server_list = compute_ps_scores( aligned_msgs_server, final_clusters_server)
    print("Final PS scores for each keyword client:\n", ps_client_list )
    print("Final PS scores for each keyword server:\n", ps_server_list )

    return pms_client, pms_server, prs, ps_client_list, ps_server_list

    


def task2_c_additions(pms, prs, ps_list):
    """
    Combine pm, pr, ps into final score.
    """
    final_scores = []
    for pm, pr, ps in zip(pms, prs, ps_list):
        final_score = (pm + pr + ps) / 3.0
        final_scores.append(final_score)
    
    # return the highest probability keyword with its index
    best_idx = np.argmax(final_scores)
    return best_idx, final_scores[best_idx]
def task2_c_multiplication(pms, prs, ps_list):
    """
    Combine pm, pr, ps into final score.
    """
    final_scores = []
    for pm, pr, ps in zip(pms, prs, ps_list):
        final_score = pm * pr * ps
        final_scores.append(final_score)
    
    # return the highest probability keyword with its index
    best_idx = np.argmax(final_scores)
    return best_idx, final_scores[best_idx]


def load_alignment_and_candidates_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["aligned"].tolist(), data["candidates"].tolist()


def convert_candidates(key_word_candidates):
    ''' convert from JSON-like structure provided by Anna to KeywordCandidate objects used in my implementation '''
        
    # conversion
    converted_candidates = [
        KeywordCandidate(
            field_id=f['field_id'],
            start_idx=f['start'],
            end_idx=f['end'],
            field_type=FieldType.STATIC if f['is_static'] else FieldType.DYNAMIC
        )
        for f in key_word_candidates
    ]

    return converted_candidates



def sheet2_task2_reverse2(args = None):
    
    # assign to variable
    if args is not None and args.max_len is not None and args.max_len > 0:
        mxLen = args.max_len
    else:
        print('Error: Max length argument is required and should be > 0')
        return


    # msgs_client_gen, unit_fields_client, merged_fields_client, keyword_candidates_client_gen = generate_dummy_data_for_task2(
    #     n_msgs=20 if TESTING else 1000,
    #     msg_len=15 if TESTING else 255,
    #     static_ratio=0.6,
    # )
    # msgs_server_gen, unit_fields_server, merged_fields_server, keyword_candidates_server_gen = generate_dummy_data_for_task2(
    #     n_msgs=20 if TESTING else 1000,
    #     msg_len=15 if TESTING else 255,
    #     static_ratio=0.6,
    # )
    msgs_client, key_word_candidates_client = load_alignment_and_candidates_npz("client_alignment_and_candidates.npz")
    msgs_server, key_word_candidates_server = load_alignment_and_candidates_npz("server_alignment_and_candidates.npz")

    
    # determine indicies of columns which contains NONE values in any row
    none_indices_client = determine_empty_indices(msgs_client)
    none_indices_server = determine_empty_indices(msgs_server)


    keyword_candidates_client = convert_candidates(key_word_candidates_client)
    keyword_candidates_server = convert_candidates(key_word_candidates_server)

    #convert msgs_client from list of lists to numpy array
    msgs_client = np.array(msgs_client)
    msgs_server = np.array(msgs_server)


    # performing Task 2.a
    filtered_candidates_client = task2_a_exclude_keywords_candidates(
        keyword_candidates_client,
        max_length=mxLen,
        none_indices=none_indices_client,
    )
    filtered_candidates_server = task2_a_exclude_keywords_candidates(
        keyword_candidates_server,
        max_length=mxLen,
        none_indices=none_indices_server,
    )

    
    # performing Task 2.b
    pms_client, pms_server, prs, ps_client_list, ps_server_list = task2_b(
        filtered_keywords_client=filtered_candidates_client,
        filtered_keywords_server=filtered_candidates_server,
        aligned_msgs_client=msgs_client,
        aligned_msgs_server=msgs_server,
    )

    # performing Task 2.c
    best_idx_client_addition, best_score_client_addition = task2_c_additions(pms_client, prs, ps_client_list)
    best_idx_server_addition, best_score_server_addition = task2_c_additions(pms_server, prs, ps_server_list)

    print(f"Best keyword client with addition final score: {best_score_client_addition}")

    print(f"Best keyword server with addition final score: {best_score_server_addition}")

    # print keywords with best idx
    if best_idx_client_addition < len(filtered_candidates_client):
        best_keyword_client = filtered_candidates_client[best_idx_client_addition]
        print(f"Best keyword client with addition: start_idx={best_keyword_client.start_idx}, end_idx={best_keyword_client.end_idx}, length={len(best_keyword_client)}")
    else:
        print("Best keyword client index is out of range of filtered candidates.")

    if best_idx_server_addition < len(filtered_candidates_server):
        best_keyword_server = filtered_candidates_server[best_idx_server_addition]
        print(f"Best keyword server with addition: start_idx={best_keyword_server.start_idx}, end_idx={best_keyword_server.end_idx}, length={len(best_keyword_server)}")
    else:
        print("Best keyword server index is out of range of filtered candidates.")


    
    # print physical bytes after best keyword candidate
    print_physical_bytes_after_keyword(
        msgs_client,
        best_keyword_client
    )
    print_physical_bytes_after_keyword(
        msgs_server,
        best_keyword_server 
    )
    return
        
# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    sheet2_task2_reverse2()