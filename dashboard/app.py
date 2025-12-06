import os
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import importlib
import importlib.metadata as _import_meta
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# When Streamlit runs this file directly ("streamlit run dashboard/app.py"),
# Python does not load the package root the same way as when importing the
# package. To make importing the sibling `utils.py` reliable, add the
# dashboard directory to sys.path and import `utils` as a local module.
import sys
import os as _os
_this_dir = _os.path.dirname(_os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import utils

# Path to cleaned data (relative to dashboard/)
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_clean', 'cleaned_purchase_data_exe.csv')

st.set_page_config(page_title="E‑commerce Customer Segmentation", layout="wide")

st.title("E‑commerce Customer Segmentation Dashboard")

@st.cache_data
def load_data(path=DATA_PATH):
    return utils.load_data(path)

@st.cache_data
def compute_rfm_and_clusters(df, n_clusters=4):
    rfm = utils.compute_rfm(df)
    if rfm is None or rfm.shape[0] == 0:
        return None, None
    kmeans, rfm_clusters = utils.cluster_rfm(rfm, n_clusters)
    return rfm, rfm_clusters

# Load
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Make sure `data_clean/cleaned_purchase_data_exe.csv` exists.")
    st.stop()

# Top metrics
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics = utils.compute_metrics(df)
with metrics_col1:
    st.metric("Total customers", metrics.get('total_customers', 0))
with metrics_col2:
    st.metric("Total revenue", f"${metrics.get('total_revenue', 0):,.2f}")
with metrics_col3:
    st.metric("Average order value", f"${metrics.get('avg_order_value', 0):,.2f}")

st.markdown("---")

# Controls
st.sidebar.header("Controls")
show_sample = st.sidebar.checkbox("Show raw sample", value=False)
n_clusters = st.sidebar.slider("KMeans clusters", min_value=2, max_value=8, value=4)
# Data / environment details in sidebar
with st.sidebar.expander("Data & environment info", expanded=True):
    # Refresh data button clears the cached loader. After clearing, ask user to reload
    if st.button("Refresh data"):
        try:
            load_data.clear()
        except Exception:
            # fallback to clearing all cached data
            try:
                st.cache_data.clear()
            except Exception:
                pass
        st.success("Cache cleared. Please reload the app in your browser to pick up new data.")

    # Current date only (no time)
    now = datetime.now().date()
    st.write("Current date:", now.strftime("%Y-%m-%d"))

    # (Data file metadata removed per user request)

    # Basic dataframe info
    try:
        st.write("Rows, columns:", df.shape)
        with st.expander("Column types", expanded=False):
            dtypes = pd.DataFrame(df.dtypes.rename('dtype'))
            st.dataframe(dtypes)
    except Exception:
        st.write("Dataframe not loaded or empty")

    # (Environment versions removed per user request)

# Time-series: monthly revenue
st.subheader("Revenue over time")
try:
    monthly = utils.monthly_revenue(df)
    fig_month = px.line(monthly, x='order_month', y='revenue', title='Monthly Revenue')
    st.plotly_chart(fig_month, use_container_width=True)
except Exception as e:
    st.info("Monthly revenue chart not available: check order_date column.")

st.markdown("---")

# Commerce insights
st.subheader("Commerce insights")
cols_ci = st.columns(2)
try:
    cat_sales = utils.category_sales(df)
    if not cat_sales.empty:
        fig_cat = px.bar(cat_sales, x='category', y='sales', title='Sales by product category')
        fig_cat.update_layout(xaxis_title='Category', yaxis_title='Sales')
        cols_ci[0].plotly_chart(fig_cat, use_container_width=True)
except Exception:
    pass

try:
    pay_stats = utils.payment_method_stats(df)
    if not pay_stats.empty:
        fig_pay = px.pie(pay_stats, names='payment_method', values='count', title='Payment method distribution')
        cols_ci[1].plotly_chart(fig_pay, use_container_width=True)
except Exception:
    pass

st.markdown("---")

# RFM + clustering (with RFM scoring)
st.subheader("Customer segments (RFM + KMeans)")
rfm, rfm_clusters = compute_rfm_and_clusters(df, n_clusters=n_clusters)
if rfm is None:
    st.info("RFM features could not be computed. Ensure your dataset has order/customer columns.")
else:
    # compute RFM scores and labels, merge into cluster table
    try:
        rfm_scores = utils.score_rfm(rfm)
        merged = rfm_clusters.merge(rfm_scores, on='customer_id', how='left')
    except Exception:
        merged = rfm_clusters

    # show cluster counts
    counts = merged['cluster'].value_counts().sort_index().reset_index()
    counts.columns = ['cluster', 'count']
    fig_counts = px.bar(counts, x='cluster', y='count', title='Customers per cluster')
    st.plotly_chart(fig_counts, use_container_width=True)

    # scatter colored by cluster, show segment on hover if available
    hover = ['customer_id'] + (['segment'] if 'segment' in merged.columns else [])
    fig_scatter = px.scatter(merged, x='frequency', y='monetary', color='cluster', hover_data=hover, title='Frequency vs Monetary by cluster')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # table & filter
    st.markdown("**Explore customers in a cluster**")
    sel_cluster = st.selectbox("Select cluster", sorted(merged['cluster'].unique()))
    subset = merged[merged['cluster'] == sel_cluster]
    display_cols = ['customer_id', 'recency', 'frequency', 'monetary']
    if 'R_score' in merged.columns:
        display_cols += ['R_score','F_score','M_score','RFM_score','segment']
    visible_df = subset[display_cols].sort_values('monetary', ascending=False).reset_index(drop=True)
    st.dataframe(visible_df)

    # Summary graphs for segments or clusters (placed directly under the table)
    group_col = 'segment' if 'segment' in merged.columns else 'cluster'
    grouped = merged.groupby(group_col).agg(
        customer_count=('customer_id', 'nunique')
    ).reset_index()

    # customers per segment/cluster (directly below the explore table)
    fig_seg_count = px.bar(grouped, x=group_col, y='customer_count', title=f'Customers per {group_col}')
    st.plotly_chart(fig_seg_count, use_container_width=True)

    # CSV download for the currently selected cluster/segment (kept after the chart)
    try:
        csv_bytes = visible_df.to_csv(index=False).encode('utf-8')
        fname = f"segment_cluster_{sel_cluster}.csv"
        st.download_button(label="Download CSV for selected cluster", data=csv_bytes, file_name=fname, mime='text/csv')
    except Exception:
        st.warning("CSV download not available for this selection.")

st.markdown("---")

# Behavior & relationships
st.subheader("Behavior & relationships")

cols_beh = st.columns(2)
try:
    cmap_time = next((c for c in df.columns if c.lower() in ['time_on_site [minutes]','time_on_site','time_spent_minutes']), None)
    cmap_clicks = next((c for c in df.columns if c.lower() in ['clicks_in_site','clicks']), None)
    cmap_value = next((c for c in df.columns if c.lower() in ['value [usd]','value usd','value_usd','price','amount','total','order_value']), None)
    if cmap_time is not None:
        fig_time = px.histogram(df, x=cmap_time, nbins=30, title='Time on site (minutes) distribution')
        cols_beh[0].plotly_chart(fig_time, use_container_width=True)
    if cmap_clicks is not None:
        fig_clicks = px.histogram(df, x=cmap_clicks, nbins=30, title='Clicks in site distribution')
        cols_beh[1].plotly_chart(fig_clicks, use_container_width=True)
except Exception:
    pass

try:
    corr_df = utils.correlation_inputs(df)
    if not corr_df.empty:
        corr = corr_df.corr()
        fig_corr = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu', origin='lower', title='Correlation heatmap')
        st.plotly_chart(fig_corr, use_container_width=True)
except Exception:
    pass

# Optional raw sample
if show_sample:
    st.subheader("Raw data sample")
    st.dataframe(df.sample(min(100, len(df))).reset_index(drop=True))

st.sidebar.markdown("---")
st.sidebar.markdown("Dashboard created by the project generator. See `reports/README.md` for report generation instructions.")
