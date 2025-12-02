import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """Load CSV and return a DataFrame. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=True, infer_datetime_format=True)
    # Try to parse common date columns
    for col in ['order_date', 'purchase_date', 'date']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df


def _column_map(df: pd.DataFrame):
    """Infer common column names used across functions from a variety of schemas.

    Returns a dict with keys: date, customer, value, category, payment, time_on_site, clicks.
    Missing keys will have value None.
    """
    cols = df.columns.str.lower().tolist()
    def first_match(options):
        for name in options:
            if name in cols:
                # return original cased name
                return df.columns[cols.index(name)]
        return None

    return {
        'date': first_match(['order_date', 'purchase_date', 'date']),
        'customer': first_match(['customer_id', 'customerid', 'customer']),
        'value': first_match(['value [usd]', 'value usd', 'value_usd', 'price', 'amount', 'total', 'order_value']),
        'category': first_match(['product_category', 'category']),
        'payment': first_match(['payment_method', 'payment', 'paymentmode']),
        'time_on_site': first_match(['time_on_site [minutes]', 'time_on_site', 'time_spent_minutes']),
        'clicks': first_match(['clicks_in_site', 'clicks'])
    }


def compute_metrics(df):
    out = {}
    # Best-effort metrics (use common column names)
    cmap = _column_map(df)
    cust_col = cmap['customer'] if cmap['customer'] else (df.columns[0] if len(df.columns) else None)
    price_col = cmap['value']
    out['total_customers'] = int(df[cust_col].nunique()) if cust_col is not None else 0
    if price_col:
        out['total_revenue'] = float(df[price_col].sum())
    else:
        out['total_revenue'] = 0.0
    if 'order_id' in df.columns and price_col:
        out['avg_order_value'] = float(df.groupby('order_id')[price_col].sum().mean())
    else:
        out['avg_order_value'] = float(df[price_col].mean()) if price_col else 0.0
    return out


def monthly_revenue(df):
    # Attempt to construct monthly revenue using a date + price column
    cmap = _column_map(df)
    date_col = cmap['date']
    price_col = cmap['value']
    if date_col is None or price_col is None:
        raise ValueError('date or price column not found')
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['order_month'] = df[date_col].dt.to_period('M').astype(str)
    monthly = df.groupby('order_month')[price_col].sum().reset_index()
    monthly = monthly.rename(columns={price_col: 'revenue'})
    return monthly


def compute_rfm(df, snapshot_date=None):
    # Compute a simple RFM using common column names
    cmap = _column_map(df)
    if cmap['customer'] is None:
        return None
    # Determine date and monetary columns
    date_col = cmap['date']
    monetary_col = cmap['value']

    # Work on a copy and ensure dates parsed if present
    df = df.copy()
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Snapshot date for recency calculation: day after last observed date
    if date_col is not None and snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Frequency: prefer unique order_id if available, otherwise count rows
    if 'order_id' in df.columns:
        frequency = df.groupby('customer_id')['order_id'].nunique()
    else:
        frequency = df.groupby('customer_id').size()

    # Recency: only if date_col exists
    if date_col is not None:
        recency = df.groupby('customer_id')[date_col].max().apply(lambda x: (snapshot_date - x).days)
    else:
        # unknown recency when dates missing
        recency = pd.Series(0, index=frequency.index)

    # Monetary
    if monetary_col:
        monetary = df.groupby('customer_id')[monetary_col].sum()
    else:
        monetary = pd.Series(0.0, index=frequency.index)

    # Combine into a DataFrame
    rfm = pd.DataFrame({
        'customer_id': frequency.index,
        'recency': recency.reindex(frequency.index).fillna(0).astype(int),
        'frequency': frequency.reindex(frequency.index).fillna(0).astype(int),
        'monetary': monetary.reindex(frequency.index).fillna(0.0).astype(float),
    }).reset_index(drop=True)
    return rfm


def cluster_rfm(rfm_df, n_clusters=4, random_state=42):
    features = rfm_df[['recency','frequency','monetary']].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    rfm_df = rfm_df.copy()
    rfm_df['cluster'] = labels
    return kmeans, rfm_df


def score_rfm(rfm_df):
    """Compute R/F/M scores (1-5) and a simple segment label.

    Returns a new DataFrame with columns: `customer_id`, `R_score`, `F_score`,
    `M_score`, `RFM_score`, `segment`.
    """
    if rfm_df is None:
        return None
    if rfm_df.shape[0] == 0:
        return rfm_df

    df = rfm_df.copy()

    # Helper to compute quintile scores (1..5). For recency, lower is better.
    def quintile_score(series, reverse=False):
        s = series.fillna(series.median())
        try:
            # qcut on the raw values; handle duplicates by ranking first
            ranks = s.rank(method='first')
            if not reverse:
                labels = [1,2,3,4,5]
            else:
                labels = [5,4,3,2,1]
            return pd.qcut(ranks, 5, labels=labels).astype(int)
        except Exception:
            # fallback: linear bins on rank
            ranks = s.rank(method='first')
            bins = np.linspace(ranks.min(), ranks.max() + 1e-9, 6)
            if not reverse:
                labels = [1,2,3,4,5]
            else:
                labels = [5,4,3,2,1]
            return pd.cut(ranks, bins=bins, labels=labels, include_lowest=True).astype(float).fillna(1).astype(int)

    # Recency: smaller is better -> reverse=True
    df['R_score'] = quintile_score(df['recency'], reverse=True)
    df['F_score'] = quintile_score(df['frequency'], reverse=False)
    df['M_score'] = quintile_score(df['monetary'], reverse=False)

    df['RFM_score'] = df[['R_score','F_score','M_score']].sum(axis=1)

    def label_from_score(score):
        # Simple thresholds; tune as needed
        if score >= 13:
            return 'Champions'
        if score >= 10:
            return 'Loyal'
        if score >= 7:
            return 'Potential'
        if score >= 5:
            return 'Needs Attention'
        return 'At Risk'

    df['segment'] = df['RFM_score'].apply(label_from_score)

    return df[['customer_id','R_score','F_score','M_score','RFM_score','segment']]


# -----------------------------
# Additional helpers for dashboard charts
# -----------------------------
def category_sales(df):
    """Return total sales per product category.
    Columns: category, sales
    """
    cmap = _column_map(df)
    cat, val = cmap['category'], cmap['value']
    if cat is None or val is None:
        return pd.DataFrame(columns=['category','sales'])
    out = df.groupby(cat)[val].sum().reset_index().rename(columns={cat:'category', val:'sales'})
    out = out.sort_values('sales', ascending=False)
    return out


def payment_method_stats(df):
    """Return counts and sales per payment method.
    Columns: payment_method, count, sales
    """
    cmap = _column_map(df)
    pay, val = cmap['payment'], cmap['value']
    if pay is None:
        return pd.DataFrame(columns=['payment_method','count','sales'])
    counts = df[pay].value_counts().rename_axis('payment_method').reset_index(name='count')
    if val is not None:
        sales = df.groupby(pay)[val].sum().reset_index().rename(columns={val:'sales', pay:'payment_method'})
        out = counts.merge(sales, on='payment_method', how='left').fillna({'sales':0})
    else:
        out = counts
        out['sales'] = 0.0
    return out


def correlation_inputs(df):
    """Return a dataframe with numeric inputs for correlation heatmap if available."""
    cmap = _column_map(df)
    cols = []
    rename = {}
    if cmap['value'] is not None:
        cols.append(cmap['value']); rename[cmap['value']] = 'value_usd'
    if cmap['time_on_site'] is not None:
        cols.append(cmap['time_on_site']); rename[cmap['time_on_site']] = 'time_on_site_min'
    if cmap['clicks'] is not None:
        cols.append(cmap['clicks']); rename[cmap['clicks']] = 'clicks'
    if not cols:
        return pd.DataFrame()
    out = df[cols].copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.rename(columns=rename)
    return out
