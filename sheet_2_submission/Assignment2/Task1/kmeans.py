from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.decomposition import PCA


GAP_VALUE = 256  # Dummy-Wert für Gaps (None), außerhalb 0..255

def cluster_single_keyword_kmedoids(aligned_messages, candidate,
                                    n_clusters=3,
                                    gap_value=GAP_VALUE,
                                    random_state=0):
    """
    Führt K-Medoids-Clustering (sklearn-extra) für GENAU EIN Keyword-Kandidaten aus.

    Parameters
    ----------
    aligned_messages : List[List[int | None]]
        Alle aligned Nachrichten (alle Zeilen gleich lang).
    candidate : dict
        Keyword-Feld mit mindestens:
            - 'field_id'
            - 'end'  (Index der letzten Spalte des Felds im Alignment)
    n_clusters : int
        Anzahl der Cluster (k in k-Medoids).
    gap_value : int
        Welcher Wert für Gaps (None) verwendet werden soll.
    random_state : int
        Seed für Reproduzierbarkeit.

    Returns
    -------
    labels : List[int]
        Cluster-Label pro Nachricht (Länge = Anzahl Nachrichten).
    """

    end = candidate["end"]

    # 1) Tail-Features bauen: alle Bytes NACH dem Feld
    tails = []
    for msg in aligned_messages:
        tail = msg[end+1:]
        # None → gap_value
        tail_clean = [gap_value if b is None else int(b) for b in tail]
        tails.append(tail_clean)

    X = np.array(tails, dtype=int)  # shape: (n_msgs, tail_len)

    # Sonderfall: hinter dem Feld kommt nichts mehr
    if X.shape[1] == 0:
        labels = np.zeros(X.shape[0], dtype=int)
        print(f"Field {candidate.get('field_id', '?')}: tail length = 0 → alle in Cluster 0.")
        return labels.tolist()

    # 2) K-Medoids mit Hamming-Distanz
    # metric='manhattan' auf ints ≈ Hamming, wenn Unterschiede selten sind,
    # sauberer ist aber: 'precomputed' mit eigener Distanzmatrix;
    # fürs Praktikum reicht 'manhattan' idR aus.
    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric="manhattan",   # approximiert Hamming auf Byte-Vektoren
        init="k-medoids++",
        random_state=random_state,
    )

    # Fit & Predict
    labels = kmedoids.fit_predict(X)

    # kleine Diagnose
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Field {candidate.get('field_id', '?')} (end={end}), k={n_clusters}")
    print("  Cluster sizes:", dict(zip(unique, counts)))

    return labels.tolist()


def extract_tail_features_for_candidate(aligned_messages, candidate, gap_value=GAP_VALUE):
    """
    Baut die Feature-Matrix X für GENAU EIN Keyword-Feld.

    aligned_messages: List[List[int | None]]
        Dein Alignment (alle Nachrichten, alle Spalten).
    candidate: dict
        Ein Eintrag aus deinen keyword_candidates mit mindestens:
            - 'end'  (Index der letzten Spalte des Felds im Alignment)

    Rückgabe:
        X: np.ndarray der Form (n_msgs, tail_len)
           → jede Zeile = Tail der Nachricht nach dem Feld.
    """
    end = candidate["end"]
    feats = []

    for msg in aligned_messages:
        # alles nach dem Keyword-Feld
        tail = msg[end+1:]
        # None → gap_value
        tail_clean = [gap_value if b is None else int(b) for b in tail]
        feats.append(tail_clean)

    return np.array(feats, dtype=int)


def plot_clusters_2d(X, labels, title="Cluster Visualization"):
    """
    X: Feature-Matrix (n_samples, n_features)
    labels: Clusterlabels (n_samples,)
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        s=20,
        alpha=0.8
    )

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)

    handles, _ = scatter.legend_elements()
    plt.legend(handles,
               [f"Cluster {i}" for i in range(len(np.unique(labels)))],
               title="Clusters")

    plt.savefig("keyword_cluster.png")


def kmeans_iat_clusters(iats, n_clusters=2):
    # 1) take only valid numeric IAT values

    # optional but recommended: reshape to (n_samples, 1) for sklearn
    X = iats.reshape(-1, 1)

    # 2) fit KMeans
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    km.fit(X)

    labels = km.labels_                  # cluster label for each IAT value
    centers = km.cluster_centers_.flatten()  # cluster centers as 1D array

    # 3) figure out which cluster is "small IAT" and which is "large IAT"
    small_cluster_id = np.argmin(centers)
    large_cluster_id = np.argmax(centers)

    # 4) a simple threshold between the two centers
    if n_clusters == 2:
        threshold = (centers[small_cluster_id] + centers[large_cluster_id]) / 2
    else:
        threshold = None  # for more than 2 clusters you'd pick differently

    return iats, labels, centers, threshold

def plot_iat_kmeans_clusters(iats, labels, centers, threshold, output_filename):
    # sort by value just for nicer plotting
    order = np.argsort(iats)
    iats_sorted = iats[order]
    labels_sorted = labels[order]

    # histogram per cluster
    unique_labels = np.unique(labels_sorted)

    plt.figure(figsize=(8, 4))
    for lab in unique_labels:
        mask = labels_sorted == lab
        plt.hist(iats_sorted[mask],
                 bins=100,
                 alpha=0.6,
                 label=f"cluster {lab} (center={centers[lab]:.4f})")

    # optional: log scale on y-axis, helps a lot
    plt.yscale("log")

    if threshold is not None:
        plt.axvline(threshold, linestyle="--", label=f"threshold ≈ {threshold:.4f}")

    plt.xlabel("IAT (seconds)")
    plt.ylabel("Frequency (log scale)")
    plt.title("KMeans clusters of IAT values")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(output_filename)



