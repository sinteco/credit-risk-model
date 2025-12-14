import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df: pd.DataFrame, customer_id_col: str, transaction_date_col: str, amount_col: str) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary (RFM) metrics.
    """
    # Ensure date column is datetime
    df[transaction_date_col] = pd.to_datetime(df[transaction_date_col])
    
    # Define snapshot date (usually max date + 1 day)
    snapshot_date = df[transaction_date_col].max() + pd.Timedelta(days=1)
    
    # Calculate RFM
    rfm = df.groupby(customer_id_col).agg({
        transaction_date_col: lambda x: (snapshot_date - x.max()).days, # Recency
        customer_id_col: 'count', # Frequency
        amount_col: 'sum' # Monetary
    }).rename(columns={
        transaction_date_col: 'Recency',
        customer_id_col: 'Frequency',
        amount_col: 'Monetary'
    })
    
    return rfm

def assign_risk_label(rfm_df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Cluster customers based on RFM and assign 'is_high_risk' label.
    High risk is typically associated with Low Frequency and Low Monetary value.
    """
    # Scale the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Identify High Risk Cluster
    # We look for the cluster with the lowest average Frequency and Monetary value
    cluster_summary = rfm_df.groupby('Cluster')[['Frequency', 'Monetary', 'Recency']].mean()
    print("Cluster Summary:")
    print(cluster_summary)
    
    # Simple heuristic: Rank clusters by Frequency + Monetary (normalized or simple sum of ranks)
    # Lower is "worse" engagement -> Higher Risk (in the context of this proxy)
    # Or, specifically for credit risk proxy in BNPL:
    # - High Recency (hasn't bought in a while) might be churned, but not necessarily default.
    # - Low Frequency/Monetary might mean new or casual users.
    # - "Disengaged" customers are labeled as high-risk proxies per instructions.
    # Instructions: "least engaged (highest-risk) customer segment (typically characterized by low frequency and low monetary value)"
    
    # We find the cluster with min(Frequency) and min(Monetary)
    # Let's score them:
    # We want the cluster with LOWEST Frequency and LOWEST Monetary.
    
    # Let's pick the cluster with the lowest mean Monetary value as the high risk one for simplicity, 
    # or a combination.
    
    high_risk_cluster = cluster_summary['Monetary'].idxmin()
    print(f"Identified Cluster {high_risk_cluster} as High Risk (Lowest Monetary Value).")
    
    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)
    
    return rfm_df

