import pandas as pd
import numpy as np
from datetime import datetime

RAW_PATH = "data/raw/online_retail.csv"
OUT_PATH = "data/processed/customer_rfm.csv"


def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove cancellations, missing customer IDs, and invalid rows."""
    df = df.copy()

    # Drop rows with no CustomerID -- can't attribute them to a segment
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove cancelled orders (InvoiceNo starting with 'C') and negative/zero quantities
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Line-item revenue
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df


def build_rfm_table(df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
    """Aggregate cleaned transactions to one row per customer: R, F, M."""
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        AvgBasketSize=("Quantity", "mean"),
        Country=("Country", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    ).reset_index()

    # Customer tenure in days (used later for CLV)
    rfm["TenureDays"] = (snapshot_date - rfm["FirstPurchase"]).dt.days.clip(lower=1)

    return rfm


def run_pipeline():
    print("Loading raw transactions...")
    raw = load_raw_data()
    print(f"  {len(raw):,} raw rows loaded")

    print("Cleaning transactions (removing cancellations, missing IDs, invalid rows)...")
    cleaned = clean_transactions(raw)
    print(f"  {len(cleaned):,} rows remain after cleaning")

    print("Building customer-level RFM table...")
    rfm = build_rfm_table(cleaned)
    print(f"  {len(rfm):,} unique customers")

    rfm.to_csv(OUT_PATH, index=False)
    print(f"Saved RFM table -> {OUT_PATH}")
    return rfm


if __name__ == "__main__":
    run_pipeline()
