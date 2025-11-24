from kmeans import cluster_single_keyword_kmedoids


def validate_cluster():


    pass


def start_clustering(keyword_candidates, alignment):
    cand = keyword_candidates[0]  # z.B. das erste Feld

    labels = cluster_single_keyword_kmedoids(
        aligned_messages=alignment,
        candidate=cand,
        n_clusters=3,
    )

    # später:
    clusters_per_field = {}
    clusters_per_field[cand["field_id"]] = labels
    return