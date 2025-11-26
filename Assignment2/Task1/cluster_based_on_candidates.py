import math
import numpy as np
import pandas as pd


#function creates clusters by values for one keyword!
#input: candidates dictionary, aligned msgs [[],[]]
#output: dictionary: {(0,23,234),[idx3,idx6,idx7,...]}
#-> key: tupel with content of candidate fields, according list of idxs that belong to that content
def cluster_by_keyword(aligned_msgs, kc):

    clusters = {}
    start = kc["start"]
    end   = kc["end"]

    for i, msg in enumerate(aligned_msgs):
        field_value = tuple(msg[start : end + 1])   #+1 weil exklusiv

        if field_value not in clusters:
            clusters[field_value] = []

        clusters[field_value].append(i)

    return clusters


def clustering_messages_by_keywords(aligned_msgs, keyword_candidates):
    print("Performing keyword-value clustering...\n")

    final_clusters = []

    for kc in keyword_candidates:
        clusters = cluster_by_keyword(aligned_msgs, kc)

        if len(clusters) <= 1:  #static keywords will only produce 1 cluster -> exclude them
            final_clusters.append(float('nan'))     #nan for static keywords!
        else:
            # sorts clusters per size internally
            clusters_sorted = dict(
                sorted(clusters.items(), key=lambda item: len(item[1]))
            )
            final_clusters.append(clusters_sorted)

    return final_clusters


#for single keyword candidate
def compute_cluster_compactness(cluster_dict, total_msgs):

    sizes = [len(idx_list) for idx_list in cluster_dict.values() if len(idx_list) > 0]

    if not sizes or total_msgs == 0:
        return {
            "num_clusters": 0,
            "num_messages": total_msgs,
            "avg_cluster_size": math.nan,
            "min_cluster_size": math.nan,
            "max_cluster_size": math.nan,
            "singleton_fraction": math.nan,
        }

    num_clusters = len(sizes)
    num_messages = total_msgs

    avg_cluster_size = num_messages / float(num_clusters)
    min_cluster_size = min(sizes)
    max_cluster_size = max(sizes)

    singletons = sum(1 for s in sizes if s == 1)
    singleton_fraction = singletons / float(num_clusters)

    return {
        "num_clusters": num_clusters,
        "num_messages": num_messages,
        "avg_cluster_size": float(avg_cluster_size),
        "min_cluster_size": int(min_cluster_size),
        "max_cluster_size": int(max_cluster_size),
        "singleton_fraction": float(singleton_fraction),
    }


def analyze_cluster_compactness(keyword_candidates, keyword_clusters, total_msgs,
                                almost_static_threshold=0.9):


    results = []

    for i, cluster_dict in enumerate(keyword_clusters):
        candidate = keyword_candidates[i]

        info = {
            "field_id": candidate.get("field_id", i),
            "start": candidate.get("start", None),
            "end": candidate.get("end", None),
        }

        is_static_flag = candidate.get("is_static", False)

        if is_static_flag:
            info.update({
                "num_clusters": 0,
                "num_messages": total_msgs,
                "avg_cluster_size": math.nan,
                "min_cluster_size": math.nan,
                "max_cluster_size": math.nan,
                "singleton_fraction": math.nan,
            })
            results.append(info)
            continue

        if not isinstance(cluster_dict, dict) or len(cluster_dict) == 0:
            info.update({
                "num_clusters": 0,
                "num_messages": total_msgs,
                "avg_cluster_size": math.nan,
                "min_cluster_size": math.nan,
                "max_cluster_size": math.nan,
                "singleton_fraction": math.nan,
            })
            results.append(info)
            continue

        sizes = [len(idx_list) for idx_list in cluster_dict.values() if len(idx_list) > 0]

        if total_msgs > 0 and sizes:
            largest_frac = max(sizes) / float(total_msgs)
        else:
            largest_frac = 0.0

        if largest_frac >= almost_static_threshold:
            info.update({
                "num_clusters": 0,
                "num_messages": total_msgs,
                "avg_cluster_size": math.nan,
                "min_cluster_size": math.nan,
                "max_cluster_size": math.nan,
                "singleton_fraction": math.nan,
            })
            results.append(info)
            continue

        metrics = compute_cluster_compactness(cluster_dict, total_msgs)
        info.update(metrics)
        results.append(info)

    return results


def display_cluster_compactness_results(results, top_n=10, sort_by="avg_cluster_size", ascending=False):

    df = pd.DataFrame(results)

    df_valid = df.dropna(subset=["avg_cluster_size"])

    df_valid["Position"] = df_valid.apply(
        lambda row: f"{int(row['start'])}:{int(row['end'])}", axis=1  # Added int() for cleaner display
    )

    if sort_by in df_valid.columns:
        df_sorted = df_valid.sort_values(by=sort_by, ascending=ascending).copy()
    else:
        df_sorted = df_valid.copy()

    cols = [
        "field_id",
        "Position",
        "num_clusters",
        "num_messages",
        "avg_cluster_size",
        "min_cluster_size",
        "max_cluster_size",
        "singleton_fraction",
    ]

    print("\n--- Cluster-Compactness-Analyse ---")
    print(df_sorted[cols].head(top_n).to_string(index=False, float_format="%.4f"))


#evaluates clusters for one keyword candidate
#input: alignment list, cluster dictionary for one keyword candidate
def evaluate_single_homogeneity(msg_lengths, cluster_dict):


    weighted_sum = 0.0
    total_weight = 0

    for message_indices in cluster_dict.values():
        if len(message_indices) < 2:
            continue

        cluster_lengths = [msg_lengths[i] for i in message_indices]

        std_dev = np.std(cluster_lengths)

        weighted_sum += std_dev * len(message_indices)
        total_weight += len(message_indices)

    if total_weight == 0:
        return float("nan")

    return float(weighted_sum / total_weight)




def analyze_payload_lengths(alignment, keyword_candidates, keyword_clusters):

    msg_lengths = [
        sum(1 for b in seq if b is not None)
        for seq in alignment
    ]

    length_analysis_results = []

    for i, cluster_result in enumerate(keyword_clusters):
        candidate = keyword_candidates[i]

        candidate_info = {
            "field_id": candidate["field_id"],
            "start": candidate["start"],
            "end": candidate["end"],
            "is_static": candidate.get("is_static", False)
        }

        #static candidates are not considered
        if candidate_info["is_static"]:
            candidate_info["homogeneity_score"] = float("nan")

        #calculate score for dynamic fields
        elif isinstance(cluster_result, dict) and len(cluster_result) > 0:
            score = evaluate_single_homogeneity(msg_lengths, cluster_result)
            candidate_info["homogeneity_score"] = score

        else:
            candidate_info["homogeneity_score"] = float("nan")

        length_analysis_results.append(candidate_info)

    return length_analysis_results



def display_payload_analysis_results(results):

    print("\n--- Analysis of payload length homogeneity ---")


    df = pd.DataFrame(results)


    df_dynamic = df.dropna(subset=['homogeneity_score'])
    df_static = df[df['homogeneity_score'].isna()]

    df_sorted = df_dynamic.sort_values(by='homogeneity_score', ascending=True)


    print("\nBest Keyword Candidates:")


    df_output = df_sorted[['field_id', 'start', 'end', 'homogeneity_score']].head(10)
    df_output['Position'] = df_output.apply(lambda row: f"{row['start']}:{row['end']}", axis=1)
    df_output = df_output.rename(columns={'homogeneity_score': 'Homogenitäts-Score (Ziel: Min. 0.0)'})

    print(df_output.to_string(index=False, float_format="%.4f"))



#idx keyword_candidates and keyword_clusters align!
#idx of sequences inside keyword clusters {(0,23,234),[idx3,idx6,idx7,...]} represent positions of sequences
def create_and_analyze_clusters(alignment, keyword_candidates):
    keyword_clusters = clustering_messages_by_keywords(alignment, keyword_candidates)
    print(keyword_clusters[1])
    results=analyze_payload_lengths(alignment, keyword_candidates, keyword_clusters)
    display_payload_analysis_results(results)


    total_msgs = len(alignment)
    compactness_results = analyze_cluster_compactness(
        keyword_candidates,
        keyword_clusters,
        total_msgs
    )
    display_cluster_compactness_results(compactness_results,
                                        top_n=10,
                                        sort_by="avg_cluster_size",
                                        ascending=False)

    return