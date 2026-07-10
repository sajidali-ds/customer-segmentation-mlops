import pandas as pd
import numpy as np
import joblib
import json

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow

from feature_engineering import run_feature_engineering, FEATURE_COLUMNS
from mlflow_tracking import (
    init_mlflow,
    log_candidate_run,
    log_and_register_best_model,
    promote_if_better,
)

MODEL_PATH = "models/segmentation_model.joblib"
META_PATH = "models/model_meta.joblib"
CENTROIDS_PATH = "models/cluster_centroids.joblib"
SEGMENTS_OUT = "data/processed/customer_segments.csv"

SEGMENT_LABELS = {
    # These get assigned dynamically based on cluster RFM profile ranking,
    # not hardcoded to a cluster id (cluster ids aren't stable across runs).
}


def find_best_kmeans(X, k_range=range(2, 9)):
    """Elbow + Silhouette search for K-Means."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil, "model": km})
    best = max(results, key=lambda r: r["silhouette"])
    return best, results


def evaluate_candidates(X):
    """Train and score every candidate algorithm; return comparison table + fitted models."""
    candidates = {}

    # 1. K-Means (best k via silhouette search)
    best_km, km_results = find_best_kmeans(X)
    candidates["KMeans"] = {
        "model": best_km["model"],
        "labels": best_km["model"].labels_,
        "k": best_km["k"],
        "silhouette": best_km["silhouette"],
        "davies_bouldin": davies_bouldin_score(X, best_km["model"].labels_),
        "params": {"n_clusters": best_km["k"], "init": "k-means++", "n_init": 10, "random_state": 42},
    }

    # 2. Gaussian Mixture Model (same k as best KMeans for fair comparison)
    gmm = GaussianMixture(n_components=best_km["k"], random_state=42, n_init=5)
    gmm_labels = gmm.fit_predict(X)
    candidates["GaussianMixture"] = {
        "model": gmm,
        "labels": gmm_labels,
        "k": best_km["k"],
        "silhouette": silhouette_score(X, gmm_labels),
        "davies_bouldin": davies_bouldin_score(X, gmm_labels),
        "params": {"n_components": best_km["k"], "n_init": 5, "random_state": 42},
    }

    # 3. Agglomerative / Hierarchical
    agg = AgglomerativeClustering(n_clusters=best_km["k"])
    agg_labels = agg.fit_predict(X)
    candidates["Agglomerative"] = {
        "model": agg,
        "labels": agg_labels,
        "k": best_km["k"],
        "silhouette": silhouette_score(X, agg_labels),
        "davies_bouldin": davies_bouldin_score(X, agg_labels),
        "params": {"n_clusters": best_km["k"], "linkage": "ward"},
    }

    # 4. DBSCAN (density-based, finds its own number of clusters + outliers)
    db = DBSCAN(eps=0.8, min_samples=10)
    db_labels = db.fit_predict(X)
    n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_clusters_db >= 2:
        # Silhouette needs to exclude noise points (-1) for a fair score
        mask = db_labels != -1
        sil_db = silhouette_score(X[mask], db_labels[mask]) if mask.sum() > 0 else -1
        dbi_db = davies_bouldin_score(X[mask], db_labels[mask]) if mask.sum() > 0 else np.inf
    else:
        sil_db, dbi_db = -1, np.inf
    candidates["DBSCAN"] = {
        "model": db,
        "labels": db_labels,
        "k": n_clusters_db,
        "silhouette": sil_db,
        "davies_bouldin": dbi_db,
        "params": {"eps": 0.8, "min_samples": 10},
    }

    return candidates


def label_segments_by_value(rfm_df: pd.DataFrame, labels: np.ndarray) -> dict:
    """
    Assign human-readable business labels to numeric cluster IDs based on
    each cluster's average Recency/Frequency/Monetary -- NOT hardcoded IDs,
    since cluster numbering is arbitrary and changes between runs.
    """
    tmp = rfm_df.copy()
    tmp["cluster"] = labels
    tmp = tmp[tmp["cluster"] != -1]  # ignore DBSCAN noise for labeling

    profile = tmp.groupby("cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    )
    # Composite value score: low recency (recent) + high frequency + high monetary = high value
    profile["value_score"] = (
        (-profile["Recency"].rank()) + profile["Frequency"].rank() + profile["Monetary"].rank()
    )
    ranked = profile.sort_values("value_score", ascending=False).index.tolist()

    label_pool = ["Champions", "Loyal Customers", "Potential Loyalists",
                  "At Risk", "Hibernating / Lost"]
    # Trim/extend label pool to match number of actual clusters found
    if len(ranked) <= len(label_pool):
        labels_assigned = label_pool[:len(ranked)]
    else:
        labels_assigned = label_pool + [f"Segment {i}" for i in range(len(ranked) - len(label_pool))]

    mapping = {cluster_id: label for cluster_id, label in zip(ranked, labels_assigned)}
    mapping[-1] = "Unclassified / Outlier"
    return mapping


def plot_pca(X, labels, segment_names, out_path="static/cluster_pca.png"):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    plot_df["Segment"] = [segment_names.get(l, str(l)) for l in labels]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="Segment", palette="Set2", s=40, alpha=0.8)
    plt.title("Customer Segments (PCA Projection)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_segment_profiles(rfm_df, labels, segment_names, out_path="static/segment_profiles.png"):
    tmp = rfm_df.copy()
    tmp["Segment"] = [segment_names.get(l, str(l)) for l in labels]
    profile = tmp.groupby("Segment")[["Recency", "Frequency", "Monetary", "CLV"]].mean()

    profile_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    profile_norm.plot(kind="bar", figsize=(9, 5))
    plt.title("Segment Profiles (Normalized Averages)")
    plt.ylabel("Normalized value (0-1)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def compute_cluster_centroids(X, labels):
    """
    Compute the mean feature vector (centroid) for every cluster in scaled
    feature space. Saved regardless of algorithm so that ANY model --
    including DBSCAN/Agglomerative, which have no native .predict() for
    new points -- can score a brand-new customer via nearest-centroid
    assignment at inference time.
    """
    centroids = {}
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        centroids[int(cluster_id)] = X[labels == cluster_id].mean(axis=0)
    return centroids


def run_training():
    tracking_uri = init_mlflow()
    print(f"MLflow tracking URI: {tracking_uri}")

    rfm, X_scaled = run_feature_engineering()

    with mlflow.start_run(run_name="segmentation_training") as parent_run:
        mlflow.log_param("n_customers", len(rfm))
        mlflow.log_param("feature_columns", ",".join(FEATURE_COLUMNS))

        print("\nEvaluating clustering algorithms...")
        candidates = evaluate_candidates(X_scaled)

        # Log EVERY candidate as its own nested MLflow run -- full experiment
        # history, not just the winner.
        comparison_rows = []
        for name, res in candidates.items():
            metrics = {
                "silhouette": res["silhouette"],
                "davies_bouldin": res["davies_bouldin"] if np.isfinite(res["davies_bouldin"]) else -1,
                "k_clusters": res["k"],
            }
            log_candidate_run(name, res["params"], metrics)
            comparison_rows.append({
                "algorithm": name,
                "k_clusters": res["k"],
                "silhouette": round(res["silhouette"], 4),
                "davies_bouldin": round(res["davies_bouldin"], 4) if np.isfinite(res["davies_bouldin"]) else None,
            })

        comparison_df = pd.DataFrame(comparison_rows).sort_values("silhouette", ascending=False)
        print("\nModel comparison (higher silhouette / lower Davies-Bouldin = better):")
        print(comparison_df.to_string(index=False))

        best_name = comparison_df.iloc[0]["algorithm"]
        best = candidates[best_name]
        print(f"\nSelected best model: {best_name} (k={best['k']}, silhouette={best['silhouette']:.4f})")

        segment_names = label_segments_by_value(rfm, best["labels"])
        rfm["cluster_id"] = best["labels"]
        rfm["segment"] = [segment_names.get(l, str(l)) for l in best["labels"]]

        rfm.to_csv(SEGMENTS_OUT, index=False)
        print(f"Saved segmented customer table -> {SEGMENTS_OUT}")

        joblib.dump(best["model"], MODEL_PATH)

        centroids = compute_cluster_centroids(X_scaled, best["labels"])
        joblib.dump(centroids, CENTROIDS_PATH)
        print(f"Saved cluster centroids -> {CENTROIDS_PATH}")

        meta = {
            "algorithm": best_name,
            "k": best["k"],
            "feature_columns": FEATURE_COLUMNS,
            "segment_names": segment_names,
            "silhouette": float(best["silhouette"]),
        }
        joblib.dump(meta, META_PATH)
        print(f"Saved model -> {MODEL_PATH}")
        print(f"Saved metadata -> {META_PATH}")

        print("Generating dashboard charts...")
        plot_pca(X_scaled, best["labels"], segment_names)
        plot_segment_profiles(rfm, best["labels"], segment_names)
        print("Saved static/cluster_pca.png and static/segment_profiles.png")

        comparison_path = "data/processed/model_comparison.json"
        comparison_df.to_json(comparison_path, orient="records", indent=2)

        # --- MLOps: log + register the winning model, then apply the
        # promotion gate (only becomes "Production" if it beats the
        # current Production model's silhouette score) ---
        final_metrics = {
            "silhouette": float(best["silhouette"]),
            "davies_bouldin": float(best["davies_bouldin"]) if np.isfinite(best["davies_bouldin"]) else -1,
            "k_clusters": best["k"],
        }
        new_version, run_id = log_and_register_best_model(
            model=best["model"],
            algorithm_name=best_name,
            params=best["params"],
            metrics=final_metrics,
            artifact_paths=[
                "static/cluster_pca.png",
                "static/segment_profiles.png",
                comparison_path,
                MODEL_PATH,
                CENTROIDS_PATH,
                META_PATH,
            ],
        )

        if new_version is not None:
            promote_if_better(new_version, float(best["silhouette"]))
        else:
            print("Warning: could not resolve new model version for registry promotion check.")

    return rfm, best, comparison_df


if __name__ == "__main__":
    run_training()
