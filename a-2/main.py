import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="number of clusters to find")
    parser.add_argument(
        "--n-clusters", type=int, help="number of features to use in a tree", default=2
    )
    parser.add_argument(
        "--data", type=str, default="data/scRNAseq_human_pancreas.csv", help="data path"
    )

    a = parser.parse_args()
    return (a.n_clusters, a.data)


def read_data(data_path):
    return anndata.read_csv(data_path)


def preprocess_data(adata: anndata.AnnData, scale: bool = True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)


def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)

    k_values = range(2, 11)

    # 2)
    print("random initialization")
    silhouettes_kmeans_random = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init=KMeans.random)
        clustering = kmeans.fit(X)

        silhouette_coef = kmeans.silhouette(clustering, X)
        silhouettes_kmeans_random.append(silhouette_coef)

        print(f"\trandom k={k}: silhouette={silhouette_coef:.6f}")

    # 3)
    print("kmeans++ initialization")
    silhouettes_kmeans_pp = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init=KMeans.pp)
        clustering = kmeans.fit(X)

        silhouette_coef = kmeans.silhouette(clustering, X)
        silhouettes_kmeans_pp.append(silhouette_coef)

        print(f"\tkmeans++ k={k}: silhouette={silhouette_coef:.6f}")

    # plotting task 2 and 3
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouettes_kmeans_random, marker="o", label="Random")
    plt.plot(k_values, silhouettes_kmeans_pp, marker="s", label="KMeans++")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Coefficient")
    plt.title("Silhouette Coefficient vs Number of Clusters")
    plt.legend()
    plt.grid(True)
    plt.savefig("silhouette_vs_k.png")

    # 4)
    best_k_random = k_values[np.argmax(silhouettes_kmeans_random)]
    best_k_pp = k_values[np.argmax(silhouettes_kmeans_pp)]

    X_2d = PCA(X=heart.X, num_components=2)
    # using best k from random
    kmeans_random = KMeans(n_clusters=best_k_random, init=KMeans.random)
    clustering_random = kmeans_random.fit(X)
    plot_title = f"Clustering with Random Initialization (k={best_k_random})"
    visualize_cluster(X_2d[:, 0], X_2d[:, 1], clustering_random, plot_title)

    # using best k from ++
    kmeans_pp = KMeans(n_clusters=best_k_pp, init=KMeans.pp)
    clustering_pp = kmeans_pp.fit(X)
    plot_title = f"Clustering with KMeans++ Initialization (k={best_k_pp})"
    visualize_cluster(X_2d[:, 0], X_2d[:, 1], clustering_pp, plot_title)


def visualize_cluster(x, y, clustering, title="Title"):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, s=10, c=clustering, cmap="plasma", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(f"{title}.png")


if __name__ == "__main__":
    main()
