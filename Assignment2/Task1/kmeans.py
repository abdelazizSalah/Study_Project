from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.decomposition import PCA


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


#helper
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



