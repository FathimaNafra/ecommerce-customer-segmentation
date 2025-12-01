

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid")


# -----------------------------
#  LOAD & CLEAN DATA
# -----------------------------
def load_and_clean(data_path):
    df = pd.read_csv(data_path)
    print("Before cleaning:", df.shape)

    # Basic cleaning
    df = df.dropna(subset=["InvoiceDate", "CustomerID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Total price
    df["Total"] = df["Quantity"] * df["Price"]

    print("After cleaning:", df.shape)
    return df


# -----------------------------
#  RFM CALCULATION
# -----------------------------
def calculate_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    RFM = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "count",
        "Total": "sum"
    })

    RFM.columns = ["Recency", "Frequency", "Monetary"]
    return RFM


# -----------------------------
#  NORMALIZE RFM
# -----------------------------
def scale_rfm(rfm):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm)
    return scaled


# -----------------------------
#  ELBOW METHOD
# -----------------------------
def elbow_plot(data):
    inertia_list = []
    K = range(1, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_list.append(kmeans.inertia_)

    plt.plot(K, inertia_list, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()


# -----------------------------
# K-MEANS CLUSTERING
# -----------------------------
def run_kmeans(data, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(data)

    score = silhouette_score(data, labels)
    print("Silhouette Score:", round(score, 3))

    return labels


# -----------------------------
#  VISUALIZATION
# -----------------------------
def plot_clusters(rfm, labels):
    rfm_plot = rfm.copy()
    rfm_plot["Cluster"] = labels

    sns.pairplot(rfm_plot, hue="Cluster", palette="viridis")
    plt.show()

    sns.boxplot(data=rfm_plot, x="Cluster", y="Monetary", palette="viridis")
    plt.show()


# -----------------------------
#  PIPELINE 
# -----------------------------
def customer_segmentation(data_path):
    df = load_and_clean(data_path)
    rfm = calculate_rfm(df)

    scaled_rfm = scale_rfm(rfm)

    print("\n📌 Showing Elbow Plot to Find Best k")
    elbow_plot(scaled_rfm)

    print("\n📌 Running KMeans with k=4 (change if needed)")
    labels = run_kmeans(scaled_rfm, k=4)

    plot_clusters(rfm, labels)

    rfm["Cluster"] = labels
    print("\nFinal RFM with Clusters:")
    print(rfm.head())

    return rfm



data_path = r"c:\Users\iyehi\OneDrive\Desktop\ecommerce.csv"
customer_segmentation(data_path)
