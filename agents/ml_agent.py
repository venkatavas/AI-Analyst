"""
ML Agent - Performs machine learning analysis using simplified implementations.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
import json
import os

class MLAgent:
    def __init__(self):
        self.scaler = StandardScaler()
    
    def analyze_ward_data(self, csv_files: List[str]) -> Dict[str, Any]:
        """Analyze ward data from multiple CSV files."""
        all_data = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                # Extract numeric features for ML analysis
                numeric_cols = ['male', 'female', 'total_illiterates_by_ward']
                available_cols = [col for col in numeric_cols if col in df.columns]
                
                if available_cols:
                    ward_data = df[['wardname'] + available_cols].copy()
                    ward_data['file_source'] = file_path
                    all_data.append(ward_data)
            except Exception as e:
                print(f"Warning: Could not process {file_path}: {e}")
        
        if not all_data:
            return {"error": "No valid data found for analysis"}
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        return {
            "total_wards": len(combined_df),
            "total_files": len(csv_files),
            "features": list(combined_df.select_dtypes(include=[np.number]).columns),
            "sample_data": combined_df.head().to_dict('records'),
            "combined_data": combined_df
        }
    
    def perform_clustering_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform KMeans clustering on ward data."""
        try:
            if "combined_data" not in data:
                return {"error": "No combined data available for clustering"}
            
            df = data["combined_data"]
            numeric_cols = ['male', 'female', 'total_illiterates_by_ward']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {"error": "Insufficient numeric features for clustering"}
            
            # Prepare features for clustering
            features = df[available_cols].fillna(0)
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform KMeans clustering
            n_clusters = min(5, len(df) // 10)  # Adaptive cluster count
            if n_clusters < 2:
                n_clusters = 2
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Add cluster labels to dataframe
            df_clustered = df.copy()
            df_clustered['cluster'] = clusters
            
            # Generate cluster summary
            cluster_summary = []
            for i in range(n_clusters):
                cluster_data = df_clustered[df_clustered['cluster'] == i]
                summary = {
                    "cluster_id": int(i),
                    "ward_count": len(cluster_data),
                    "avg_illiterates": float(cluster_data['total_illiterates_by_ward'].mean()),
                    "top_wards": cluster_data.nlargest(3, 'total_illiterates_by_ward')['wardname'].tolist()
                }
                cluster_summary.append(summary)
            
            return {
                "status": "success",
                "algorithm": "KMeans",
                "n_clusters": n_clusters,
                "total_wards_analyzed": len(df),
                "cluster_summary": cluster_summary,
                "features_used": available_cols
            }
            
        except Exception as e:
            return {"error": f"Clustering analysis failed: {str(e)}"}
    
    def perform_anomaly_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Isolation Forest anomaly detection."""
        try:
            if "combined_data" not in data:
                return {"error": "No combined data available for anomaly detection"}
            
            df = data["combined_data"]
            numeric_cols = ['male', 'female', 'total_illiterates_by_ward']
            available_cols = [col for col in numeric_cols if col in df.columns]
            
            if len(available_cols) < 2:
                return {"error": "Insufficient numeric features for anomaly detection"}
            
            # Prepare features
            features = df[available_cols].fillna(0)
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(features_scaled)
            
            # Identify anomalies
            df_anomalies = df.copy()
            df_anomalies['is_anomaly'] = anomaly_labels == -1
            
            anomalous_wards = df_anomalies[df_anomalies['is_anomaly']]
            
            # Generate anomaly summary
            anomaly_summary = []
            for _, ward in anomalous_wards.iterrows():
                summary = {
                    "ward_name": ward['wardname'],
                    "total_illiterates": int(ward['total_illiterates_by_ward']),
                    "male": int(ward['male']),
                    "female": int(ward['female']),
                    "anomaly_score": float(isolation_forest.decision_function(
                        self.scaler.transform([ward[available_cols].values]))[0])
                }
                anomaly_summary.append(summary)
            
            return {
                "status": "success",
                "algorithm": "Isolation Forest",
                "total_wards_analyzed": len(df),
                "anomalous_wards_count": len(anomalous_wards),
                "anomaly_percentage": round((len(anomalous_wards) / len(df)) * 100, 2),
                "anomalous_wards": anomaly_summary,
                "features_used": available_cols
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
