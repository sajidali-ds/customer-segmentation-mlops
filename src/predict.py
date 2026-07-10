import numpy as np
import joblib
import pandas as pd

MODEL_PATH = "models/segmentation_model.joblib"
SCALER_PATH = "models/scaler.joblib"
META_PATH = "models/model_meta.joblib"
CENTROIDS_PATH = "models/cluster_centroids.joblib"


class SegmentPredictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.meta = joblib.load(META_PATH)
        self.centroids = joblib.load(CENTROIDS_PATH)
        self.segment_names = self.meta["segment_names"]
        self.algorithm = self.meta["algorithm"]

    def _build_feature_row(self, recency, frequency, monetary, tenure_days, avg_basket_size):
        clv = (monetary / max(tenure_days, 1)) * 365
        row = pd.DataFrame([{
            "Recency": recency,
            "Frequency": frequency,
            "Monetary": monetary,
            "CLV": clv,
            "AvgBasketSize": avg_basket_size,
        }])
        # Same log-transform as training for Monetary & CLV
        row["Monetary"] = np.log1p(row["Monetary"])
        row["CLV"] = np.log1p(row["CLV"].clip(lower=0))
        return row, clv

    def predict(self, recency, frequency, monetary, tenure_days, avg_basket_size):
        row, clv = self._build_feature_row(recency, frequency, monetary, tenure_days, avg_basket_size)
        X_scaled = self.scaler.transform(row)

        if self.algorithm in ("KMeans", "GaussianMixture"):
            cluster_id = int(self.model.predict(X_scaled)[0])
        else:
            # DBSCAN / Agglomerative don't support .predict on new points natively.
            # Assign to the nearest saved cluster centroid instead -- this is the
            # standard, correct way to score new points for these algorithms.
            cluster_id = self._nearest_centroid(X_scaled[0])

        segment_name = self.segment_names.get(cluster_id, self.segment_names.get(str(cluster_id), "Unknown"))

        return {
            "cluster_id": cluster_id,
            "segment": segment_name,
            "estimated_clv": round(float(clv), 2),
            "algorithm_used": self.algorithm,
        }

    def _nearest_centroid(self, x_scaled_row):
        """Assign a new point to whichever saved cluster centroid it's closest to
        (Euclidean distance in scaled feature space). Used for DBSCAN/Agglomerative,
        which have no native .predict() for out-of-sample points."""
        best_cluster, best_dist = None, np.inf
        for cluster_id, centroid in self.centroids.items():
            dist = np.linalg.norm(x_scaled_row - centroid)
            if dist < best_dist:
                best_dist, best_cluster = dist, cluster_id
        return int(best_cluster)


if __name__ == "__main__":
    predictor = SegmentPredictor()
    example = predictor.predict(recency=5, frequency=18, monetary=2400, tenure_days=300, avg_basket_size=6.2)
    print("Example prediction:", example)
