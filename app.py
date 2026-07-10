import os
import sys
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import SegmentPredictor  # noqa: E402

app = Flask(__name__)

SEGMENTS_CSV = "data/processed/customer_segments.csv"
META_PATH = "models/model_meta.joblib"
COMPARISON_JSON = "data/processed/model_comparison.json"

_predictor = None


def get_predictor():
    """Lazy-load the model so `python app.py` gives a friendly error if train.py hasn't run yet."""
    global _predictor
    if _predictor is None:
        _predictor = SegmentPredictor()
    return _predictor


def _models_are_trained():
    return os.path.exists(SEGMENTS_CSV) and os.path.exists(META_PATH)


@app.route("/")
def dashboard():
    if not _models_are_trained():
        return (
            "<h2>Model not trained yet.</h2>"
            "<p>Run the pipeline first:</p>"
            "<pre>python src/generate_sample_data.py\n"
            "python src/data_pipeline.py\n"
            "python src/train.py</pre>",
            200,
        )

    df = pd.read_csv(SEGMENTS_CSV)
    meta = joblib.load(META_PATH)

    segment_summary = (
        df.groupby("segment")
        .agg(
            customers=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            avg_clv=("CLV", "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values("avg_monetary", ascending=False)
    )

    return render_template(
        "index.html",
        segments=segment_summary.to_dict(orient="records"),
        algorithm=meta["algorithm"],
        k=meta["k"],
        silhouette=round(meta["silhouette"], 3),
        total_customers=len(df),
    )


@app.route("/api/segments", methods=["GET"])
def api_segments():
    if not _models_are_trained():
        return jsonify({"error": "Model not trained. Run train.py first."}), 400

    df = pd.read_csv(SEGMENTS_CSV)
    summary = (
        df.groupby("segment")
        .agg(
            customers=("CustomerID", "count"),
            avg_recency_days=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            avg_clv=("CLV", "mean"),
        )
        .round(2)
        .reset_index()
    )
    return jsonify(summary.to_dict(orient="records"))


@app.route("/api/customers", methods=["GET"])
def api_customers():
    if not _models_are_trained():
        return jsonify({"error": "Model not trained. Run train.py first."}), 400

    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 25))
    segment_filter = request.args.get("segment")

    df = pd.read_csv(SEGMENTS_CSV)
    if segment_filter:
        df = df[df["segment"] == segment_filter]

    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return jsonify({
        "total": len(df),
        "page": page,
        "page_size": page_size,
        "customers": page_df[["CustomerID", "Recency", "Frequency", "Monetary", "CLV", "segment"]].to_dict(orient="records"),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expected JSON body:
    {
        "recency": 12,
        "frequency": 8,
        "monetary": 950.5,
        "tenure_days": 180,
        "avg_basket_size": 4.5
    }
    """
    if not _models_are_trained():
        return jsonify({"error": "Model not trained. Run train.py first."}), 400

    data = request.get_json(force=True)
    required = ["recency", "frequency", "monetary", "tenure_days", "avg_basket_size"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        predictor = get_predictor()
        result = predictor.predict(
            recency=float(data["recency"]),
            frequency=float(data["frequency"]),
            monetary=float(data["monetary"]),
            tenure_days=float(data["tenure_days"]),
            avg_basket_size=float(data["avg_basket_size"]),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    if not _models_are_trained():
        return jsonify({"error": "Model not trained. Run train.py first."}), 400
    meta = joblib.load(META_PATH)
    comparison = []
    if os.path.exists(COMPARISON_JSON):
        comparison = pd.read_json(COMPARISON_JSON).to_dict(orient="records")
    return jsonify({
        "selected_algorithm": meta["algorithm"],
        "k_clusters": meta["k"],
        "silhouette_score": meta["silhouette"],
        "segment_names": meta["segment_names"],
        "all_algorithms_compared": comparison,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_trained": _models_are_trained()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
