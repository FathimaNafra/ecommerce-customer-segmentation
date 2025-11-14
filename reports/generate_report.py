import os
import pandas as pd
import plotly.express as px

# Location of cleaned data
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_clean', 'cleaned_purchase_data_exe.csv')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report.html')


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=True, infer_datetime_format=True)
    for col in ['order_date', 'purchase_date', 'date']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
    return df


def make_report(df):
    # Basic metrics
    total_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 'n/a'
    price_col = next((c for c in ['price','amount','total','order_value'] if c in df.columns), None)
    total_revenue = df[price_col].sum() if price_col else 0

    # Monthly revenue
    date_col = next((c for c in ['order_date','purchase_date','date'] if c in df.columns), None)
    if date_col and price_col:
        df['order_month'] = pd.to_datetime(df[date_col], errors='coerce').dt.to_period('M').astype(str)
        monthly = df.groupby('order_month')[price_col].sum().reset_index().rename(columns={price_col:'revenue'})
        fig_month = px.line(monthly, x='order_month', y='revenue', title='Monthly Revenue')
        month_html = fig_month.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        month_html = '<p>No monthly revenue chart (missing date or price column)</p>'

    html = f"""
    <html>
    <head><title>Report</title></head>
    <body>
      <h1>E‑commerce Report</h1>
      <p>Total customers: {total_customers}</p>
      <p>Total revenue: ${total_revenue:,.2f}</p>
      <h2>Monthly revenue</h2>
      {month_html}
    </body>
    </html>
    """
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report written to {OUTPUT}")


if __name__ == '__main__':
    df = load_data()
    make_report(df)
