"""
Simple ML Agent - Pure Python implementation without scikit-learn dependencies
Provides basic clustering and anomaly detection using native algorithms
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import json
import math
import random

class SimpleMlAgent:
    def __init__(self):
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
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
            "combined_data": combined_df
        }
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def simple_kmeans(self, data, k=5, max_iterations=100):
        """Simple K-means clustering implementation."""
        if len(data) < k:
            k = len(data)
        
        # Initialize centroids randomly
        centroids = random.sample(data, k)
        
        for iteration in range(max_iterations):
            # Assign points to clusters
            clusters = [[] for _ in range(k)]
            assignments = []
            
            for point in data:
                distances = [self.euclidean_distance(point, centroid) for centroid in centroids]
                closest_cluster = distances.index(min(distances))
                clusters[closest_cluster].append(point)
                assignments.append(closest_cluster)
            
            # Update centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    # Calculate mean of cluster points
                    centroid = [sum(coord) / len(cluster) for coord in zip(*cluster)]
                    new_centroids.append(centroid)
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(centroids[len(new_centroids)])
            
            # Check for convergence
            if centroids == new_centroids:
                break
            
            centroids = new_centroids
        
        return assignments, centroids
    
    def detect_anomalies(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect anomalies using IQR-based outlier detection."""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Focus on total illiterates column
            if 'total_illiterates_by_ward' not in df.columns:
                return {
                    'status': 'error',
                    'message': 'Required column total_illiterates_by_ward not found'
                }
            
            illiterates = df['total_illiterates_by_ward']
            
            # Calculate IQR-based outlier detection
            Q1 = illiterates.quantile(0.25)
            Q3 = illiterates.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier threshold
            threshold = Q3 + 1.5 * IQR
            mean_val = illiterates.mean()
            
            # Identify outliers
            outliers = df[df['total_illiterates_by_ward'] > threshold]
            
            # Calculate statistics
            anomaly_list = []
            for _, row in outliers.iterrows():
                deviation = ((row['total_illiterates_by_ward'] - mean_val) / mean_val) * 100
                anomaly_list.append({
                    'ward': row.get('wardname', 'Unknown'),
                    'illiterates': int(row['total_illiterates_by_ward']),
                    'deviation_percent': f"+{deviation:.0f}%",
                    'risk_category': 'critical' if deviation > 200 else 'high'
                })
            
            return {
                'status': 'success',
                'summary': {
                    'total_outliers': len(outliers),
                    'detection_method': 'IQR',
                    'threshold': int(threshold),
                    'mean_deviation': f"+{((outliers['total_illiterates_by_ward'].mean() - mean_val) / mean_val * 100):.0f}%" if len(outliers) > 0 else "0%"
                },
                'anomalies': anomaly_list
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Anomaly detection failed: {str(e)}"
            }

    def perform_clustering_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform simple K-means clustering on data."""
        try:
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data)
            
            # Check if this is governance data (ward-based) or general data
            if 'total_illiterates_by_ward' in df.columns:
                return self._cluster_governance_data(df)
            else:
                return self._cluster_general_data(df)
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Clustering analysis failed: {str(e)}"
            }
    
    def _cluster_governance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster governance datasets with ward-level data."""
        try:
            # Use multiple features for clustering
            feature_cols = ['male', 'female', 'total_illiterates_by_ward']
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                return {
                    'status': 'error',
                    'message': 'No suitable columns found for clustering'
                }
            
            # Extract features and convert to list of lists
            features = df[available_cols].values.tolist()
            
            # Perform k-means clustering (k=5)
            k = 5
            assignments, centroids = self.simple_kmeans(features, k)
            
            # Analyze clusters
            cluster_summary = {}
            for i in range(k):
                cluster_indices = [j for j, assignment in enumerate(assignments) if assignment == i]
                if cluster_indices:
                    cluster_wards = df.iloc[cluster_indices]
                    avg_illiterates = cluster_wards['total_illiterates_by_ward'].mean()
                    
                    # Get top wards safely
                    if 'wardname' in cluster_wards.columns and len(cluster_wards) > 0:
                        # Sort by total_illiterates_by_ward and get top wards
                        sorted_wards = cluster_wards.sort_values('total_illiterates_by_ward', ascending=False)
                        top_wards = sorted_wards['wardname'].head(3).tolist()
                    else:
                        top_wards = ['Unknown']
                    
                    cluster_summary[f'cluster_{i+1}'] = {
                        'wards': len(cluster_wards),
                        'avg_illiterates': int(avg_illiterates),
                        'top_wards': top_wards[:2],
                        'risk_level': 'high' if avg_illiterates > 2000 else 'medium' if avg_illiterates > 1000 else 'low'
                    }
            
            # Calculate coefficient of variation
            cv = (df['total_illiterates_by_ward'].std() / df['total_illiterates_by_ward'].mean()) * 100
            
            return {
                'status': 'success',
                'algorithm': 'Simple K-Means',
                'n_clusters': k,
                'total_wards_analyzed': len(df),
                'cluster_summary': [
                    {
                        'cluster_id': i,
                        'ward_count': cluster_summary[f'cluster_{i+1}']['wards'],
                        'avg_illiterates': cluster_summary[f'cluster_{i+1}']['avg_illiterates'],
                        'top_wards': cluster_summary[f'cluster_{i+1}']['top_wards'],
                        'risk_level': cluster_summary[f'cluster_{i+1}']['risk_level']
                    } for i in range(k) if f'cluster_{i+1}' in cluster_summary
                ],
                'coefficient_of_variation': round(cv, 2)
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Governance clustering failed: {str(e)}"
            }
    
    def _cluster_general_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster general datasets (non-governance data like Skill Development)."""
        try:
            # For general datasets, perform basic statistical analysis instead of clustering
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return {
                    'status': 'success',
                    'message': 'No numeric columns found for clustering - performed basic analysis',
                    'analysis_type': 'statistical_summary',
                    'columns_analyzed': list(df.columns),
                    'total_records': len(df)
                }
            
            # Create statistical summary
            stats_summary = {}
            for col in numeric_cols:
                stats_summary[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            
            return {
                'status': 'success',
                'analysis_type': 'statistical_summary',
                'message': 'General dataset analyzed - clustering not applicable',
                'columns_analyzed': numeric_cols,
                'total_records': len(df),
                'statistical_summary': stats_summary
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f"General data analysis failed: {str(e)}"
            }
    
    def simple_outlier_detection(self, data, threshold_factor=2.0):
        """Simple outlier detection using statistical methods."""
        outliers = []
        
        for i, point in enumerate(data):
            # Calculate distance to all other points
            distances = [self.euclidean_distance(point, other) for j, other in enumerate(data) if i != j]
            
            if distances:
                avg_distance = sum(distances) / len(distances)
                std_distance = math.sqrt(sum((d - avg_distance) ** 2 for d in distances) / len(distances))
                
                # Mark as outlier if average distance is significantly higher
                if avg_distance > (sum(distances) / len(distances)) + threshold_factor * std_distance:
                    outliers.append(i)
        
        return outliers
    
    def perform_anomaly_detection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform simple outlier detection."""
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
            
            # Normalize features
            normalized_features = []
            for col in available_cols:
                col_data = features[col]
                min_val, max_val = col_data.min(), col_data.max()
                if max_val > min_val:
                    normalized = [(x - min_val) / (max_val - min_val) for x in col_data]
                else:
                    normalized = [0.5] * len(col_data)
                normalized_features.append(normalized)
            
            # Convert to list of points
            data_points = list(zip(*normalized_features))
            
            # Detect outliers
            outlier_indices = self.simple_outlier_detection(data_points)
            
            # Generate anomaly summary
            anomaly_summary = []
            for idx in outlier_indices:
                ward = df.iloc[idx]
                summary = {
                    "ward_name": ward['wardname'],
                    "total_illiterates": int(ward['total_illiterates_by_ward']),
                    "male": int(ward['male']),
                    "female": int(ward['female']),
                    "anomaly_score": float(len(outlier_indices) / len(df))  # Simple score
                }
                anomaly_summary.append(summary)
            
            return {
                "status": "success",
                "algorithm": "Statistical Outlier Detection",
                "total_wards_analyzed": len(df),
                "anomalous_wards_count": len(outlier_indices),
                "anomaly_percentage": round((len(outlier_indices) / len(df)) * 100, 2),
                "anomalous_wards": anomaly_summary,
                "features_used": available_cols
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}

# Alias for compatibility
MLAgent = SimpleMlAgent
