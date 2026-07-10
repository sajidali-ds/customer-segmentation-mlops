import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

IN_PATH = "data/processed/customer_rfm.csv"
OUT_PATH = "data/processed/customer_features.csv"
SCALER_PATH = "models/scaler.joblib"

FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary", "CLV", "AvgBasketSize"]


def add_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Quintile-based R/F/M scores (5 = best), the classic marketing RFM score."""
    df = rfm.copy()

    # Recency: lower is better -> reverse the quantile labels
    df["R_Score"] = pd.qcut(df["Recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["RFM_Score"] = df["R_Score"] + df["F_Score"] + df["M_Score"]

    return df


def add_clv_estimate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple historical CLV proxy:
        CLV = (Monetary / TenureDays) * 365
    i.e. annualized spend rate. Good enough for segmentation purposes;
    swap in BG/NBD + Gamma-Gamma (lifetimes library) for a probabilistic
    CLV model in a more advanced iteration.
    """
    df = df.copy()
    df["CLV"] = (df["Monetary"] / df["TenureDays"]) * 365
    return df


def scale_features(df: pd.DataFrame, fit: bool = True):
    """Standard-scale the numeric features used for clustering."""
    X = df[FEATURE_COLUMNS].copy()

    # Log-transform monetary/CLV (heavily right-skewed) before scaling
    X["Monetary"] = np.log1p(X["Monetary"])
    X["CLV"] = np.log1p(X["CLV"].clip(lower=0))

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def run_feature_engineering():
    print("Loading RFM table...")
    rfm = pd.read_csv(IN_PATH)

    print("Adding CLV estimate...")
    rfm = add_clv_estimate(rfm)

    print("Adding RFM quantile scores...")
    rfm = add_rfm_scores(rfm)

    rfm.to_csv(OUT_PATH, index=False)
    print(f"Saved engineered feature table -> {OUT_PATH}")

    print("Scaling features for clustering...")
    X_scaled, scaler = scale_features(rfm, fit=True)
    print(f"Feature matrix shape: {X_scaled.shape}")

    return rfm, X_scaled


if __name__ == "__main__":
    run_feature_engineering()
