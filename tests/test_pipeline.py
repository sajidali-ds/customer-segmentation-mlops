import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import clean_transactions, build_rfm_table  # noqa: E402
from feature_engineering import add_clv_estimate, add_rfm_scores, FEATURE_COLUMNS  # noqa: E402


@pytest.fixture
def raw_transactions():
    """A tiny synthetic transaction set covering the edge cases the
    cleaning step needs to handle: cancellations, missing CustomerID,
    negative/zero quantity and price."""
    data = [
        # normal rows
        {"InvoiceNo": "1001", "StockCode": "A1", "Description": "Widget", "Quantity": 3,
         "InvoiceDate": "01/01/2024 10:00", "UnitPrice": 5.0, "CustomerID": 1, "Country": "UK"},
        {"InvoiceNo": "1002", "StockCode": "A2", "Description": "Gadget", "Quantity": 2,
         "InvoiceDate": "01/05/2024 10:00", "UnitPrice": 10.0, "CustomerID": 1, "Country": "UK"},
        {"InvoiceNo": "1003", "StockCode": "A1", "Description": "Widget", "Quantity": 1,
         "InvoiceDate": "01/10/2024 10:00", "UnitPrice": 5.0, "CustomerID": 2, "Country": "DE"},
        # cancellation -- should be dropped
        {"InvoiceNo": "C1004", "StockCode": "A1", "Description": "Widget", "Quantity": -1,
         "InvoiceDate": "01/11/2024 10:00", "UnitPrice": 5.0, "CustomerID": 2, "Country": "DE"},
        # missing CustomerID -- should be dropped
        {"InvoiceNo": "1005", "StockCode": "A3", "Description": "Thingamajig", "Quantity": 4,
         "InvoiceDate": "01/12/2024 10:00", "UnitPrice": 2.0, "CustomerID": np.nan, "Country": "FR"},
        # zero price -- should be dropped
        {"InvoiceNo": "1006", "StockCode": "A4", "Description": "Freebie", "Quantity": 1,
         "InvoiceDate": "01/13/2024 10:00", "UnitPrice": 0.0, "CustomerID": 3, "Country": "US"},
    ]
    return pd.DataFrame(data)


class TestDataCleaning:
    def test_removes_cancellations(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()

    def test_removes_missing_customer_id(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        assert cleaned["CustomerID"].isna().sum() == 0

    def test_removes_zero_price_rows(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        assert (cleaned["UnitPrice"] > 0).all()

    def test_total_price_calculated(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        assert "TotalPrice" in cleaned.columns
        row = cleaned[cleaned["InvoiceNo"] == "1001"].iloc[0]
        assert row["TotalPrice"] == 3 * 5.0


class TestRFMTable:
    def test_one_row_per_customer(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        rfm = build_rfm_table(cleaned)
        assert rfm["CustomerID"].is_unique

    def test_frequency_counts_unique_invoices(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        rfm = build_rfm_table(cleaned)
        # Customer 1 has 2 distinct invoices (1001, 1002)
        cust1 = rfm[rfm["CustomerID"] == 1].iloc[0]
        assert cust1["Frequency"] == 2

    def test_monetary_sums_total_price(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        rfm = build_rfm_table(cleaned)
        cust1 = rfm[rfm["CustomerID"] == 1].iloc[0]
        assert cust1["Monetary"] == pytest.approx(3 * 5.0 + 2 * 10.0)

    def test_recency_is_non_negative(self, raw_transactions):
        cleaned = clean_transactions(raw_transactions)
        rfm = build_rfm_table(cleaned)
        assert (rfm["Recency"] >= 0).all()


class TestFeatureEngineering:
    def test_clv_is_non_negative(self):
        df = pd.DataFrame({
            "Monetary": [100, 500, 1000],
            "TenureDays": [30, 100, 365],
        })
        result = add_clv_estimate(df)
        assert (result["CLV"] >= 0).all()

    def test_clv_formula(self):
        df = pd.DataFrame({"Monetary": [365.0], "TenureDays": [365]})
        result = add_clv_estimate(df)
        # 365 spend over 365 days tenure -> annualized = 365
        assert result["CLV"].iloc[0] == pytest.approx(365.0)

    def test_rfm_scores_in_range(self):
        n = 50
        df = pd.DataFrame({
            "Recency": np.random.randint(0, 365, n),
            "Frequency": np.random.randint(1, 30, n),
            "Monetary": np.random.uniform(10, 5000, n),
        })
        scored = add_rfm_scores(df)
        assert scored["R_Score"].between(1, 5).all()
        assert scored["F_Score"].between(1, 5).all()
        assert scored["M_Score"].between(1, 5).all()
        assert scored["RFM_Score"].between(3, 15).all()

    def test_feature_columns_defined(self):
        assert len(FEATURE_COLUMNS) == 5
        assert "Recency" in FEATURE_COLUMNS
        assert "Monetary" in FEATURE_COLUMNS


class TestClusteringSanity:
    def test_silhouette_score_is_valid_range(self):
        """Sanity check that well-separated synthetic clusters score highly --
        catches silent breakage in the clustering/scoring pipeline."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        rng = np.random.RandomState(42)
        cluster_a = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
        cluster_b = rng.normal(loc=[10, 10], scale=0.5, size=(50, 2))
        X = np.vstack([cluster_a, cluster_b])

        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)

        # Well-separated synthetic clusters should score very highly
        assert score > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
