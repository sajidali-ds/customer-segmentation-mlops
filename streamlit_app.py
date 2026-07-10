import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

SEGMENTS_CSV = "data/processed/customer_segments.csv"
META_PATH = "models/model_meta.joblib"

st.set_page_config(
    page_title="Customer Segmentation Explorer",
    page_icon="📊",
    layout="wide",
)

SEGMENT_COLORS = {
    "Champions": "#1f6f5c",
    "Loyal Customers": "#3f8f78",
    "Potential Loyalists": "#6fb98f",
    "At Risk": "#d98c3f",
    "Hibernating / Lost": "#b5502f",
    "Unclassified / Outlier": "#9aa39c",
}


def models_are_trained():
    return os.path.exists(SEGMENTS_CSV) and os.path.exists(META_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(SEGMENTS_CSV)
    return df


@st.cache_resource
def load_predictor():
    from predict import SegmentPredictor
    return SegmentPredictor()


# ---------------------------------------------------------------------
# Guard: friendly message if the pipeline hasn't been run yet
# ---------------------------------------------------------------------
if not models_are_trained():
    st.title("📊 Customer Segmentation Explorer")
    st.warning(
        "No trained model found.\n\n"
        "Run the following commands from the project directory:\n\n"
        "```bash\n"
        "python src/data_pipeline.py\n"
        "python src/train.py\n"
        "```\n\n"
        "Then refresh this page."
    )
    st.stop()

    
df = load_data()
meta = joblib.load(META_PATH)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("📊 Customer Segmentation Explorer")
st.caption(
    f"Model: **{meta['algorithm']}** · {meta['k']} segments · "
    f"Silhouette score: **{meta['silhouette']:.3f}** · "
    f"{len(df):,} customers segmented"
)

tab1, tab2, tab3 = st.tabs(["🏠 Overview", "🔮 Predict a Segment", "🔍 Explore Customers"])

# ---------------------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------------------
with tab1:
    st.subheader("Segment sizes & value")

    summary = (
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

    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        color = SEGMENT_COLORS.get(row["segment"], "#1f6f5c")
        with col:
            st.markdown(
                f"""
                <div style="border:1px solid #dfe4e0; border-radius:8px; padding:16px; background:#fff;">
                    <div style="font-weight:600; color:{color};">{row['segment']}</div>
                    <div style="font-size:1.8rem; font-weight:700;">{int(row['customers'])}</div>
                    <div style="font-size:0.8rem; color:#666;">customers</div>
                    <hr style="margin:8px 0;">
                    <div style="font-size:0.8rem;">Avg spend: <b>${row['avg_monetary']:,.0f}</b></div>
                    <div style="font-size:0.8rem;">Avg CLV/yr: <b>${row['avg_clv']:,.0f}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Segment distribution")
    c1, c2 = st.columns(2)

    with c1:
        fig_pie = px.pie(
            summary, values="customers", names="segment",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            hole=0.45, title="Customers per segment"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        fig_bar = px.bar(
            summary, x="segment", y="avg_clv",
            color="segment", color_discrete_map=SEGMENT_COLORS,
            title="Average annual CLV by segment", text_auto=".2s"
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Recency vs Frequency vs Monetary (all customers)")
    fig_scatter = px.scatter(
        df, x="Recency", y="Frequency", size="Monetary", color="segment",
        color_discrete_map=SEGMENT_COLORS, hover_data=["CustomerID", "CLV"],
        opacity=0.7, title="Each dot = one customer"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 2: Predict a segment for a new/hypothetical customer
# ---------------------------------------------------------------------
with tab2:
    st.subheader("Score a new customer")
    st.caption("Move the sliders to describe a customer and see which segment they fall into.")

    c1, c2 = st.columns(2)
    with c1:
        recency = st.slider("Recency — days since last order", 0, 365, 15)
        frequency = st.slider("Frequency — number of orders", 1, 50, 10)
        monetary = st.slider("Monetary — total amount spent ($)", 10, 10000, 800)
    with c2:
        tenure_days = st.slider("Tenure — days as a customer", 1, 730, 200)
        avg_basket_size = st.slider("Avg items per order", 1, 20, 5)

    if st.button("🔮 Predict Segment", type="primary"):
        predictor = load_predictor()
        result = predictor.predict(
            recency=recency, frequency=frequency, monetary=monetary,
            tenure_days=tenure_days, avg_basket_size=avg_basket_size,
        )
        color = SEGMENT_COLORS.get(result["segment"], "#1f6f5c")
        st.markdown(
            f"""
            <div style="border-left: 6px solid {color}; padding:16px 20px; background:#f6f7f5; border-radius:6px;">
                <div style="font-size:0.85rem; color:#666;">Predicted segment</div>
                <div style="font-size:1.6rem; font-weight:700; color:{color};">{result['segment']}</div>
                <div style="margin-top:8px; font-size:0.9rem;">
                    Estimated annual CLV: <b>${result['estimated_clv']:,.2f}</b><br>
                    Model used: <code>{result['algorithm_used']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------
# TAB 3: Explore / filter the actual customer table
# ---------------------------------------------------------------------
with tab3:
    st.subheader("Browse segmented customers")

    segments_available = ["All"] + sorted(df["segment"].unique().tolist())
    chosen_segment = st.selectbox("Filter by segment", segments_available)

    filtered = df if chosen_segment == "All" else df[df["segment"] == chosen_segment]

    st.write(f"Showing **{len(filtered):,}** customers")
    st.dataframe(
        filtered[["CustomerID", "Recency", "Frequency", "Monetary", "CLV", "segment"]]
        .sort_values("Monetary", ascending=False)
        .reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download this segment as CSV",
        data=csv,
        file_name=f"customers_{chosen_segment.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )
