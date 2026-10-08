from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

from scipy.cluster.hierarchy import linkage, dendrogram


# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# Features used for clustering
FEATURES = [
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]


def load_data():
    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    # Convert college tier such as "Tier 1" into 1
    if (
        "college_tier" in df.columns
        and not pd.api.types.is_numeric_dtype(df["college_tier"])
    ):
        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)")
        )

    df_subset = df[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    scaler = StandardScaler()

    X = scaler.fit_transform(df_subset)

    return X


def calculate_scores(X, labels):
    # Ignore DBSCAN noise points
    mask = labels != -1

    X_clean = X[mask]
    labels_clean = labels[mask]

    number_of_clusters = len(set(labels_clean))

    if number_of_clusters < 2:
        return np.nan, np.nan, number_of_clusters

    sample_size = min(len(X_clean), 2000)

    silhouette = silhouette_score(
        X_clean,
        labels_clean,
        sample_size=sample_size,
        random_state=42
    )

    davies_bouldin = davies_bouldin_score(
        X_clean,
        labels_clean
    )

    return (
        round(silhouette, 4),
        round(davies_bouldin, 4),
        number_of_clusters
    )


def run():

    print("Loading data...")

    X = load_data()

    n_samples = len(X)

    print(f"Dataset loaded with {n_samples} rows.")
    print("Running K-Means evaluation...")

    # --------------------------------------------------
    # K-Means evaluation
    # --------------------------------------------------

    sample_size = min(n_samples, 2000)

    k_values = range(2, 11)

    inertias = []
    silhouette_values = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=42
        )

        labels = model.fit_predict(X)

        inertias.append(model.inertia_)

        silhouette = silhouette_score(
            X,
            labels,
            sample_size=sample_size,
            random_state=42
        )

        silhouette_values.append(silhouette)

        print(f"Finished K={k}")

    best_k = list(k_values)[
        int(np.argmax(silhouette_values))
    ]

    print(
        "\nOptimal K selected by highest silhouette:",
        best_k
    )

    # --------------------------------------------------
    # Elbow curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.xlabel("K")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve - Placement Dataset")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUT / "clustering_kmeans_elbow.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Silhouette curve
    # --------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_values,
        marker="o"
    )

    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis - Placement Dataset")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUT / "clustering_kmeans_silhouette.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # Model comparisons
    # --------------------------------------------------

    print("\nRunning model comparisons...")

    # --------------------------------------------------
    # K-Means
    # --------------------------------------------------

    kmeans = KMeans(
        n_clusters=best_k,
        n_init=10,
        random_state=42
    )

    kmeans_labels = kmeans.fit_predict(X)

    km_silhouette, km_db, km_clusters = calculate_scores(
        X,
        kmeans_labels
    )

    # --------------------------------------------------
    # Agglomerative Clustering
    # Use 5000 samples because 100000 rows
    # require too much memory
    # --------------------------------------------------

    print("Running Agglomerative Clustering...")

    cluster_sample_size = min(n_samples, 5000)

    rng = np.random.RandomState(42)

    sample_indices = rng.choice(
        n_samples,
        cluster_sample_size,
        replace=False
    )

    X_cluster = X[sample_indices]

    agglomerative = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )

    agglomerative_labels = agglomerative.fit_predict(
        X_cluster
    )

    ag_silhouette, ag_db, ag_clusters = calculate_scores(
        X_cluster,
        agglomerative_labels
    )

    # --------------------------------------------------
    # Hierarchical Dendrogram
    # --------------------------------------------------

    print("Generating dendrogram...")

    Z = linkage(
        X_cluster,
        method="ward"
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30
    )

    plt.title(
        "Hierarchical Dendrogram - Placement Dataset"
    )

    plt.xlabel("Cluster / Sample Index")
    plt.ylabel("Distance")

    plt.tight_layout()

    plt.savefig(
        OUT / "hierarchical_dendrogram.png",
        dpi=150
    )

    plt.close()

    # --------------------------------------------------
    # DBSCAN
    # --------------------------------------------------

    print("Running DBSCAN...")

    dbscan = DBSCAN(
        eps=0.8,
        min_samples=5
    )

    dbscan_labels = dbscan.fit_predict(X)

    db_silhouette, db_db, db_clusters = calculate_scores(
        X,
        dbscan_labels
    )

    noise_points = int(
        np.sum(dbscan_labels == -1)
    )

    # --------------------------------------------------
    # Comparison table
    # --------------------------------------------------

    results = pd.DataFrame(
        [
            [
                "K-Means",
                km_clusters,
                km_silhouette,
                km_db,
                0
            ],
            [
                "Agglomerative",
                ag_clusters,
                ag_silhouette,
                ag_db,
                0
            ],
            [
                "DBSCAN",
                db_clusters,
                db_silhouette,
                db_db,
                noise_points
            ]
        ],
        columns=[
            "Algorithm",
            "Clusters",
            "Silhouette",
            "Davies_Bouldin",
            "Noise_Points"
        ]
    )

    print("\nClustering Comparison:")

    print(
        results.to_string(index=False)
    )

    # Save comparison table

    results.to_csv(
        OUT / "clustering_comparison.csv",
        index=False
    )

    print(
        f"\nAll outputs and plots saved to: {OUT}"
    )


if __name__ == "__main__":
    run()