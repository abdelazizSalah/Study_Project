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
)


# -----------------------------
# Tasks
# -----------------------------
def task2_a_exclude_long_keyword_candidates(
    keyword_candidates: List[KeywordCandidate],
    max_length: int = 4,
) -> List[KeywordCandidate]:
    """
    Exclude keyword candidates longer than max_length.
    """
    print(f"Original candidates count: {len(keyword_candidates)}, and max_length: {max_length}")
   
    filtered_candidates = [
        kc for kc in keyword_candidates # +1, if only 1 element then si = 2 ei = 2, then len = 2-2 + 1 = 1
        if (kc.end_idx - kc.start_idx + 1) <= max_length #TODO: I should handle the case of zero length
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


def sheet2_task2_reverse2(args):
    
    # # read the max_length from the arguments
    # # define the argument parser
    # parser = argparse.ArgumentParser()

    # # define the argument letter, its type, and its default value
    # parser.add_argument('-m', '--max_len', type=int, default=4, help='Maximum length of keyword candidates to include')
    
    # # read it from the command line
    # args = parser.parse_args()


    
    # assign to variable
    mxLen = args.max_len

    msgs_client, unit_fields_client, merged_fields_client, keyword_candidates_client = generate_dummy_data_for_task2(
        n_msgs=20 if TESTING else 1000,
        msg_len=15 if TESTING else 255,
        static_ratio=0.6,
    )
    msgs_server, unit_fields_server, merged_fields_server, keyword_candidates_server = generate_dummy_data_for_task2(
        n_msgs=20 if TESTING else 1000,
        msg_len=15 if TESTING else 255,
        static_ratio=0.6,
    )

    # keyword_candidates = keyword_candidates_client + keyword_candidates_server
    # msgs = np.vstack((msgs_client, msgs_server))
    # print(f'msgs shape: {msgs.shape}')
    # performing Task 2.a
    filtered_candidates_client = task2_a_exclude_long_keyword_candidates(
        keyword_candidates_client,
        max_length=mxLen,
    )
    filtered_candidates_server = task2_a_exclude_long_keyword_candidates(
        keyword_candidates_server,
        max_length=mxLen,
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






# -----------------------------
# Example run
# -----------------------------
# if __name__ == "__main__":
#     sheet2_task2_reverse2()