from Assignment2.Task1.kmeans import cluster_single_keyword_kmedoids
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
        # value als tuple, damit es dictionary-key werden kann
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


def analyze_amount_and_size():

    return


#evaluates clusters for one keyword candidate
#input: alignment list, cluster dictionary for one keyword candidate
def evaluate_single_homogeneity(msg_lengths, cluster_dict):
    """
    Misst die Homogenität der Nachrichtenlängen innerhalb der Cluster
    eines Kandidaten. msg_lengths[i] ist die effektive Länge der i-ten
    Nachricht (ohne Gaps).

    Rückgabe: gewichteter Mittelwert der Standardabweichungen
    über alle Cluster (je kleiner, desto homogener).
    """

    weighted_sum = 0.0
    total_weight = 0

    for message_indices in cluster_dict.values():
        # Mindestens 2 Nachrichten für eine sinnvolle Std-Abweichung
        if len(message_indices) < 2:
            continue

        # Längen der Nachrichten in diesem Cluster
        cluster_lengths = [msg_lengths[i] for i in message_indices]

        std_dev = np.std(cluster_lengths)

        # Gewichtung nach Clustergröße
        weighted_sum += std_dev * len(message_indices)
        total_weight += len(message_indices)

    if total_weight == 0:
        return float("nan")

    return float(weighted_sum / total_weight)




def analyze_payload_lengths(alignment, keyword_candidates, keyword_clusters):
    """
    Berechnet für jeden Keyword-Kandidaten einen Homogenitäts-Score
    basierend auf den effektiven Längen der Nachrichten (ohne Gaps).

    - alignment: Liste der ausgerichteten Nachrichten (mit Gaps = None)
    - keyword_candidates: Liste der Kandidaten-Dicts
    - keyword_clusters: Liste der Cluster-Dicts (pro Kandidat)

    Rückgabe: Liste von Dicts mit u.a. 'field_id', 'start', 'end',
              'is_static', 'homogeneity_score'.
    """

    # 1) Effektive Längen (ohne Gaps) vorab berechnen
    #    Falls deine Gaps anders kodiert sind, hier anpassen.
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

        # Statische Felder nicht über Längen-Homogenität bewerten
        if candidate_info["is_static"]:
            candidate_info["homogeneity_score"] = float("nan")

        # Dynamische Felder mit Cluster-Dict -> Score berechnen
        elif isinstance(cluster_result, dict) and len(cluster_result) > 0:
            score = evaluate_single_homogeneity(msg_lengths, cluster_result)
            candidate_info["homogeneity_score"] = score

        # Falls irgendwas schief ist -> als unbrauchbar markieren
        else:
            candidate_info["homogeneity_score"] = float("nan")

        length_analysis_results.append(candidate_info)

    return length_analysis_results



def display_payload_analysis_results(results):
    """
    Sortiert und zeigt die Ergebnisse der Payload-Längen-Analyse an.
    """
    if not results:
        print("Keine Ergebnisse zur Anzeige vorhanden.")
        return

    print("\n--- Analyse der Payload-Längen-Homogenität ---")

    # 1. Konvertierung in Pandas DataFrame
    df = pd.DataFrame(results)

    # 2. Statische Felder herausfiltern (NaNs)
    df_dynamic = df.dropna(subset=['homogeneity_score'])
    df_static = df[df['homogeneity_score'].isna()]

    print(f"Es wurden {len(df_static)} statische oder unbrauchbare Kandidaten (Score=NaN) ignoriert.")

    if df_dynamic.empty:
        print("Keine dynamischen Kandidaten gefunden, die zur Analyse geeignet wären.")
        return

    # 3. Sortierung nach dem Homogenitäts-Score (aufsteigend: niedrigster Score ist bester)
    # Der niedrigste Score bedeutet die geringste Varianz der Paketlänge im Cluster.
    df_sorted = df_dynamic.sort_values(by='homogeneity_score', ascending=True)

    # 4. Ausgabe der Top 10 Kandidaten
    print("\nTop 10 Keyword-Kandidaten (sortiert nach Längen-Homogenität):")

    # Formatierung für bessere Lesbarkeit
    df_output = df_sorted[['field_id', 'start', 'end', 'homogeneity_score']].head(10)
    df_output['Position'] = df_output.apply(lambda row: f"{row['start']}:{row['end']}", axis=1)
    df_output = df_output.rename(columns={'homogeneity_score': 'Homogenitäts-Score (Ziel: Min. 0.0)'})

    # Ausgabe der Tabelle
    print(df_output.to_string(index=False, float_format="%.4f"))

    print("\nDer Kandidat in der ersten Zeile ist das wahrscheinlichste Keyword (z.B. der Statuscode).")



def print_cluster_lengths_for_candidate(alignment, keyword_candidates, keyword_clusters, candidate_idx):
    """
    Zeigt für einen Keyword-Kandidaten alle Cluster an,
    wobei nur die effektiven Längen der Nachrichten (ohne Gaps) ausgegeben werden.
    """

    candidate = keyword_candidates[candidate_idx]
    clusters = keyword_clusters[candidate_idx]

    # Effektive Längen (Gaps ignorieren: None -> nicht mitzählen)
    msg_lengths = [
        sum(1 for b in seq if b is not None)
        for seq in alignment
    ]

    print(f"\n### Keyword-Kandidat #{candidate_idx}: "
          f"{candidate['field_id']} [{candidate['start']}:{candidate['end']}] "
          f"- {len(clusters)} Value-Cluster ###\n")

    for cluster_idx, (value, msg_indices) in enumerate(clusters.items()):
        # Längen für alle Nachrichten in diesem Cluster
        lengths = [msg_lengths[i] for i in msg_indices]

        if not lengths:
            continue

        lengths_arr = np.array(lengths, dtype=float)

        print("=" * 100)
        print(f"Cluster {cluster_idx} – Feldwert: {value}  ->  {len(msg_indices)} Nachrichten")
        print(f"Längen (ohne Gaps): {lengths}")
        print(f"  min:   {lengths_arr.min():.0f}")
        print(f"  max:   {lengths_arr.max():.0f}")
        print(f"  mean:  {lengths_arr.mean():.2f}")
        print(f"  std:   {lengths_arr.std():.2f}")
        print()


#idx keyword_candidates and keyword_clusters align!
#idx of sequences inside keyword clusters {(0,23,234),[idx3,idx6,idx7,...]} represent positions of sequences
def create_and_analyze_clusters(alignment, keyword_candidates):
    keyword_clusters = clustering_messages_by_keywords(alignment, keyword_candidates)
    print(keyword_clusters[1])
    results=analyze_payload_lengths(alignment, keyword_candidates, keyword_clusters)
    display_payload_analysis_results(results)
    #print(keyword_clusters[])
    print_cluster_lengths_for_candidate(
        alignment=alignment,
        keyword_candidates=keyword_candidates,
        keyword_clusters=keyword_clusters,
        candidate_idx=1,
    )

    return