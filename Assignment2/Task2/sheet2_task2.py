TESTING = True
import argparse
import numpy as np
from typing import List

from task2_utilities import (
    KeywordCandidate,
    generate_dummy_data_for_task2,
    compute_similarity_matrix,
    clustering_messages_by_keywords,
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
    Exclude keyword candidates longer than max_length.
    """
    print(f"Original candidates count: {len(keyword_candidates)}, and max_length: {max_length}")
    print(f"None indices: {none_indices}")
    filtered_candidates = [
        # also exclude those candidates which have None values in their range
        kc for kc in keyword_candidates # +1, if only 1 element then si = 2 ei = 2, then len = 2-2 + 1 = 1
        if (kc.end_idx - kc.start_idx + 1) <= max_length and not any(idx in none_indices for idx in range(kc.start_idx, kc.end_idx + 1)) 
    ]
    print(f"Filtered candidates count: {len(filtered_candidates)}")
    return filtered_candidates




def task2_b(filtered_keywords_client: List[KeywordCandidate], filtered_keywords_server: List[KeywordCandidate], aligned_msgs_client, aligned_msgs_server=None):
    
    sim_mat = compute_similarity_matrix(aligned_msgs_client)
    final_clusters_client = clustering_messages_by_keywords(
        aligned_msgs=aligned_msgs_client,
        filtered_keywords=filtered_keywords_client,
    )
    final_clusters_server = clustering_messages_by_keywords(
        aligned_msgs=aligned_msgs_server,
        filtered_keywords=filtered_keywords_server,
    )
    pms_client = compute_message_similarity_scores(sim_mat, final_clusters_client)
    print("Final PM scores for each keyword client:\n", pms_client )

    pms_server = compute_message_similarity_scores(sim_mat, final_clusters_server)
    print("Final PM scores for each keyword server:\n", pms_server )

    prs = compute_prs(final_clusters_client, final_clusters_server, aligned_msgs_client, aligned_msgs_server)
    print("Final PR scores for each keyword:\n", prs )

    ps_client_list = compute_ps_scores(filtered_keywords_client, aligned_msgs_client)
    ps_server_list = compute_ps_scores(filtered_keywords_server, aligned_msgs_server)
    print("Final PS scores for each keyword client:\n", ps_client_list )
    print("Final PS scores for each keyword server:\n", ps_server_list )

    return pms_client, pms_server, prs, ps_client_list, ps_server_list

    


def task2_c(pms, prs, ps_list):
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


def load_alignment_and_candidates_npz(filepath):
    data = np.load(filepath, allow_pickle=True)
    return data["aligned"].tolist(), data["candidates"].tolist()


def convert_candidates(key_word_candidates):
        
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
    mxLen = args.max_len if args is not None else 4

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
    best_idx_client, best_score_client = task2_c(pms_client, prs, ps_client_list)
    best_idx_server, best_score_server = task2_c(pms_server, prs, ps_server_list)

    print(f"Best keyword client index: {best_idx_client} with final score: {best_score_client}")

    print(f"Best keyword server index: {best_idx_server} with final score: {best_score_server}")

    # print keywords with best idx
    if best_idx_client < len(filtered_candidates_client):
        best_keyword_client = filtered_candidates_client[best_idx_client]
        print(f"Best keyword client: start_idx={best_keyword_client.start_idx}, end_idx={best_keyword_client.end_idx}, length={len(best_keyword_client)}")
    else:
        print("Best keyword client index is out of range of filtered candidates.")

    if best_idx_server < len(filtered_candidates_server):
        best_keyword_server = filtered_candidates_server[best_idx_server]
        print(f"Best keyword server: start_idx={best_keyword_server.start_idx}, end_idx={best_keyword_server.end_idx}, length={len(best_keyword_server)}")
    else:
        print("Best keyword server index is out of range of filtered candidates.")

    # print the bytes in the aligned msgs which lie after the end_idx of the best keyword
    if best_idx_client < len(filtered_candidates_client):
        print_physical_bytes_after_keyword(msgs_client, best_keyword_client)

    if best_idx_server < len(filtered_candidates_server):
        print_physical_bytes_after_keyword(msgs_server, best_keyword_server)
        
# -----------------------------
# Example run
# -----------------------------
if __name__ == "__main__":
    sheet2_task2_reverse2()