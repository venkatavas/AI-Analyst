#!/usr/bin/env python3
"""
ML Agent for RTGS CLI - Simplified Version
Handles machine learning tasks including clustering and anomaly detection
using direct scikit-learn implementations.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Simplified version without CrewAI
CREWAI_AVAILABLE = False
print("🔧 Using direct ML implementation (CrewAI disabled for compatibility)")

def analyze_ward_data(csv_files: List[str]) -> Dict[str, Any]:
    """Analyze ward data from multiple CSV files to extract features for ML analysis."""
    combined_data = []
    ward_info = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            file_name = Path(csv_file).stem
            
            if 'wardname' in df.columns:
                ward_groups = df.groupby('wardname').agg({
                    'male': 'sum',
                    'female': 'sum',
                    'transgender': 'sum' if 'transgender' in df.columns else lambda x: 0
                }).reset_index()
                
                for _, row in ward_groups.iterrows():
                    total = row['male'] + row['female'] + row.get('transgender', 0)
                    if total > 0:
                        features = {
                            'total_illiterates': total,
                            'male_ratio': row['male'] / total,
                            'female_ratio': row['female'] / total,
                            'transgender_ratio': row.get('transgender', 0) / total,
                            'gender_disparity': abs(row['female'] - row['male']) / total,
                            'male_count': row['male'],
                            'female_count': row['female'],
                            'dataset': file_name
                        }
                        
                        combined_data.append([
                            features['total_illiterates'],
                            features['male_ratio'],
                            features['female_ratio'],
                            features['gender_disparity'],
                            features['male_count'],
                            features['female_count']
                        ])
                        
                        ward_info.append({
                            'ward_name': row['wardname'],
                            'dataset': file_name,
                            'features': features
                        })
        
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    return {
        'combined_data': combined_data,
        'ward_info': ward_info
    }

def perform_clustering_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform K-means clustering on ward data."""
    combined_data = data['combined_data']
    ward_info = data['ward_info']
    
    if not combined_data:
        return {"error": "No valid data found for clustering"}
    
    # Convert to numpy array and standardize
    X = np.array(combined_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal number of clusters
    optimal_k = min(8, max(2, len(ward_info)//3))
    
    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels) if optimal_k > 1 else 0
    
    # Organize results by clusters
    clusters = {}
    for i, (ward, label) in enumerate(zip(ward_info, cluster_labels)):
        cluster_id = f"cluster_{label}"
        if cluster_id not in clusters:
            clusters[cluster_id] = {
                "cluster_id": label,
                "wards": [],
                "characteristics": {}
            }
        
        clusters[cluster_id]["wards"].append({
            "ward_name": ward['ward_name'],
            "dataset": ward['dataset'],
            "total_illiterates": ward['features']['total_illiterates'],
            "male_ratio": round(ward['features']['male_ratio'], 3),
            "female_ratio": round(ward['features']['female_ratio'], 3),
            "gender_disparity": round(ward['features']['gender_disparity'], 3)
        })
    
    # Calculate cluster characteristics
    for cluster_id, cluster_data in clusters.items():
        wards = cluster_data["wards"]
        if wards:
            avg_illiterates = np.mean([w['total_illiterates'] for w in wards])
            avg_male_ratio = np.mean([w['male_ratio'] for w in wards])
            avg_female_ratio = np.mean([w['female_ratio'] for w in wards])
            avg_disparity = np.mean([w['gender_disparity'] for w in wards])
            
            cluster_data["characteristics"] = {
                "avg_total_illiterates": round(avg_illiterates, 1),
                "avg_male_ratio": round(avg_male_ratio, 3),
                "avg_female_ratio": round(avg_female_ratio, 3),
                "avg_gender_disparity": round(avg_disparity, 3),
                "ward_count": len(wards),
                "datasets_represented": list(set(w['dataset'] for w in wards))
            }
    
    return {
        "total_wards": len(ward_info),
        "num_clusters": optimal_k,
        "silhouette_score": round(silhouette_avg, 3),
        "clusters": clusters
    }

def perform_anomaly_detection(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform anomaly detection on ward data using Isolation Forest."""
    combined_data = data['combined_data']
    ward_info = data['ward_info']
    
    if not combined_data:
        return {"error": "No valid data found for anomaly detection"}
    
    # Convert to numpy array and standardize
    X = np.array(combined_data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform anomaly detection
    isolation_forest = IsolationForest(
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_estimators=100
    )
    
    anomaly_labels = isolation_forest.fit_predict(X_scaled)
    anomaly_scores = isolation_forest.decision_function(X_scaled)
    
    # Identify anomalies
    anomalies = []
    
    for i, (ward, is_anomaly, score) in enumerate(zip(ward_info, anomaly_labels, anomaly_scores)):
        if is_anomaly == -1:  # Anomaly detected
            anomaly_type = _classify_anomaly_type(ward['features'], combined_data)
            anomalies.append({
                "ward_name": ward['ward_name'],
                "dataset": ward['dataset'],
                "total_illiterates": ward['features']['total_illiterates'],
                "male_count": ward['features']['male_count'],
                "female_count": ward['features']['female_count'],
                "male_ratio": round(ward['features']['male_ratio'], 3),
                "female_ratio": round(ward['features']['female_ratio'], 3),
                "gender_disparity": round(ward['features']['gender_disparity'], 3),
                "anomaly_score": round(float(score), 4),
                "anomaly_type": anomaly_type,
                "anomaly_reason": _explain_anomaly(ward['features'], anomaly_type)
            })
    
    # Sort anomalies by score (most anomalous first)
    anomalies.sort(key=lambda x: x['anomaly_score'])
    
    # Calculate statistics
    total_wards = len(ward_info)
    anomaly_count = len(anomalies)
    anomaly_percentage = (anomaly_count / total_wards) * 100 if total_wards > 0 else 0
    
    return {
        "total_wards_analyzed": total_wards,
        "anomalies_detected": anomaly_count,
        "anomaly_percentage": round(anomaly_percentage, 1),
        "anomalies": anomalies[:10],  # Top 10 most anomalous
        "summary_statistics": {
            "avg_total_illiterates": round(np.mean([w['features']['total_illiterates'] for w in ward_info]), 1),
            "avg_gender_disparity": round(np.mean([w['features']['gender_disparity'] for w in ward_info]), 3),
            "datasets_analyzed": list(set(w['dataset'] for w in ward_info))
        }
    }

def _classify_anomaly_type(features: Dict, all_data: List) -> str:
    """Classify the type of anomaly based on features."""
    total = features['total_illiterates']
    gender_disparity = features['gender_disparity']
    
    # Calculate percentiles for comparison
    all_totals = [row[0] for row in all_data]
    all_disparities = [row[3] for row in all_data]
    
    total_percentile = np.percentile(all_totals, 90)
    disparity_percentile = np.percentile(all_disparities, 90)
    
    if total > total_percentile:
        return "high_illiteracy"
    elif total < np.percentile(all_totals, 10):
        return "low_illiteracy"
    elif gender_disparity > disparity_percentile:
        return "high_gender_disparity"
    elif gender_disparity < np.percentile(all_disparities, 10):
        return "low_gender_disparity"
    else:
        return "statistical_outlier"

def _explain_anomaly(features: Dict, anomaly_type: str) -> str:
    """Provide human-readable explanation for the anomaly."""
    explanations = {
        "high_illiteracy": f"Exceptionally high illiteracy count ({features['total_illiterates']}) compared to other wards",
        "low_illiteracy": f"Unusually low illiteracy count ({features['total_illiterates']}) compared to other wards",
        "high_gender_disparity": f"Significant gender disparity ({features['gender_disparity']:.3f}) in illiteracy rates",
        "low_gender_disparity": f"Unusually balanced gender distribution ({features['gender_disparity']:.3f}) in illiteracy",
        "statistical_outlier": "Statistical pattern differs significantly from typical wards"
    }
    return explanations.get(anomaly_type, "Anomalous pattern detected in illiteracy data")

class MLAgent:
    """ML Agent using CrewAI for advanced analytics."""
    
    def __init__(self):
        self.hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        if CREWAI_AVAILABLE:
            # Create specialized agents for different ML tasks
            self.data_analyst = Agent(
                role='Data Analyst',
                goal='Analyze ward-level illiteracy data and extract meaningful patterns',
                backstory="""You are an expert data analyst specializing in governance and demographic data. 
                You excel at identifying patterns in illiteracy data across different wards and districts.""",
                verbose=True,
                allow_delegation=False
            )
            
            self.clustering_specialist = Agent(
                role='Clustering Specialist',
                goal='Group wards with similar illiteracy patterns using advanced clustering techniques',
                backstory="""You are a machine learning specialist focused on unsupervised learning. 
                You use K-means clustering and other techniques to identify groups of wards with similar characteristics.""",
                verbose=True,
                allow_delegation=False,
                tools=[analyze_ward_data, perform_clustering_analysis]
            )
            
            self.anomaly_detector = Agent(
                role='Anomaly Detection Specialist',
                goal='Identify unusual patterns and outliers in ward-level illiteracy data',
                backstory="""You are an expert in anomaly detection and statistical outlier identification. 
                You use Isolation Forest and other techniques to find wards that deviate significantly from normal patterns.""",
                verbose=True,
                allow_delegation=False,
                tools=[analyze_ward_data, perform_anomaly_detection]
            )
    
    def cluster_wards(self, csv_files: List[str]) -> Dict[str, Any]:
        """Cluster wards across multiple CSV files using CrewAI."""
        if not CREWAI_AVAILABLE:
            return self._fallback_cluster_wards(csv_files)
        
        # Create clustering task
        clustering_task = Task(
            description=f"""
            Analyze ward-level illiteracy data from {len(csv_files)} CSV files and perform clustering analysis.
            
            Steps:
            1. Load and analyze data from all CSV files: {csv_files}
            2. Extract ward-level features including total illiterates, gender ratios, and disparities
            3. Perform K-means clustering to group wards with similar patterns
            4. Calculate cluster characteristics and silhouette scores
            5. Generate comprehensive clustering report with insights
            
            Focus on identifying meaningful patterns that can inform governance decisions.
            """,
            expected_output="Detailed clustering analysis with ward groups, characteristics, and methodology",
            agent=self.clustering_specialist
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[self.clustering_specialist],
            tasks=[clustering_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            
            # Parse the crew result and format as expected
            data = analyze_ward_data(csv_files)
            clustering_results = perform_clustering_analysis(data)
            
            return {
                "clustering_results": clustering_results,
                "methodology": {
                    "algorithm": "K-Means with CrewAI orchestration",
                    "features_used": ["total_illiterates", "male_ratio", "female_ratio", "gender_disparity"],
                    "preprocessing": "StandardScaler normalization",
                    "ai_framework": "CrewAI multi-agent system"
                },
                "crew_insights": str(result)
            }
            
        except Exception as e:
            print(f"CrewAI clustering failed: {e}, using fallback")
            return self._fallback_cluster_wards(csv_files)
    
    def detect_anomalies(self, csv_files: List[str]) -> Dict[str, Any]:
        """Detect anomalous wards using CrewAI."""
        if not CREWAI_AVAILABLE:
            return self._fallback_detect_anomalies(csv_files)
        
        # Create anomaly detection task
        anomaly_task = Task(
            description=f"""
            Analyze ward-level illiteracy data from {len(csv_files)} CSV files and detect anomalous patterns.
            
            Steps:
            1. Load and analyze data from all CSV files: {csv_files}
            2. Extract ward-level features and statistical patterns
            3. Apply Isolation Forest algorithm to identify outliers
            4. Classify anomaly types (high/low illiteracy, gender disparities, etc.)
            5. Generate detailed anomaly report with explanations
            
            Focus on identifying wards that require immediate attention or further investigation.
            """,
            expected_output="Comprehensive anomaly detection report with flagged wards and explanations",
            agent=self.anomaly_detector
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[self.anomaly_detector],
            tasks=[anomaly_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            result = crew.kickoff()
            
            # Parse the crew result and format as expected
            data = analyze_ward_data(csv_files)
            anomaly_results = perform_anomaly_detection(data)
            
            return {
                "anomaly_detection_results": anomaly_results,
                "methodology": {
                    "algorithm": "Isolation Forest with CrewAI orchestration",
                    "contamination_rate": 0.1,
                    "features_used": ["total_illiterates", "gender_ratios", "gender_disparity"],
                    "ai_framework": "CrewAI multi-agent system"
                },
                "crew_insights": str(result)
            }
            
        except Exception as e:
            print(f"CrewAI anomaly detection failed: {e}, using fallback")
            return self._fallback_detect_anomalies(csv_files)
    
    def _fallback_cluster_wards(self, csv_files: List[str]) -> Dict[str, Any]:
        """Fallback clustering without CrewAI."""
        data = analyze_ward_data(csv_files)
        clustering_results = perform_clustering_analysis(data)
        
        return {
            "clustering_results": clustering_results,
            "methodology": {
                "algorithm": "K-Means (fallback mode)",
                "features_used": ["total_illiterates", "male_ratio", "female_ratio", "gender_disparity"],
                "preprocessing": "StandardScaler normalization",
                "ai_framework": "Direct scikit-learn implementation"
            }
        }
    
    def _fallback_detect_anomalies(self, csv_files: List[str]) -> Dict[str, Any]:
        """Fallback anomaly detection without CrewAI."""
        data = analyze_ward_data(csv_files)
        anomaly_results = perform_anomaly_detection(data)
        
        return {
            "anomaly_detection_results": anomaly_results,
            "methodology": {
                "algorithm": "Isolation Forest (fallback mode)",
                "contamination_rate": 0.1,
                "features_used": ["total_illiterates", "gender_ratios", "gender_disparity"],
                "ai_framework": "Direct scikit-learn implementation"
            }
        }

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Hugging Face sentence transformers."""
        if not self.hf_api_key:
            print("Warning: No Hugging Face API key found, using fallback embeddings")
            return self._fallback_embeddings(texts)
        
        try:
            # Use Hugging Face Inference API for embeddings
            api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
            
            response = requests.post(
                api_url,
                headers=self.hf_headers,
                json={"inputs": texts},
                timeout=30
            )
            
            if response.status_code == 200:
                embeddings = response.json()
                return np.array(embeddings)
            else:
                print(f"HF API error: {response.status_code}, using fallback")
                return self._fallback_embeddings(texts)
                
        except Exception as e:
            print(f"Error generating embeddings: {e}, using fallback")
            return self._fallback_embeddings(texts)
    
    def _fallback_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback embedding generation using simple features."""
        embeddings = []
        for text in texts:
            # Simple feature extraction
            features = [
                len(text),
                text.count(' '),
                text.count(','),
                len(set(text.lower().split())),
                hash(text.lower()) % 1000 / 1000.0
            ]
            embeddings.append(features)
        return np.array(embeddings)
    
    def cluster_wards(self, csv_files: List[str]) -> Dict[str, Any]:
        """Cluster wards across multiple CSV files based on illiteracy patterns."""
        print("Loading and processing datasets for clustering...")
        
        # Combine data from all CSV files
        combined_data = []
        ward_info = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_name = Path(csv_file).stem
                
                # Extract ward-level features
                if 'wardname' in df.columns:
                    ward_groups = df.groupby('wardname').agg({
                        'male': 'sum',
                        'female': 'sum',
                        'transgender': 'sum' if 'transgender' in df.columns else lambda x: 0
                    }).reset_index()
                    
                    for _, row in ward_groups.iterrows():
                        total = row['male'] + row['female'] + row.get('transgender', 0)
                        if total > 0:
                            features = {
                                'total_illiterates': total,
                                'male_ratio': row['male'] / total,
                                'female_ratio': row['female'] / total,
                                'transgender_ratio': row.get('transgender', 0) / total,
                                'gender_disparity': abs(row['female'] - row['male']) / total,
                                'dataset': file_name
                            }
                            
                            combined_data.append(list(features.values())[:-1])  # Exclude dataset name from features
                            ward_info.append({
                                'ward_name': row['wardname'],
                                'dataset': file_name,
                                'features': features
                            })
            
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if not combined_data:
            return {"error": "No valid data found for clustering"}
        
        # Convert to numpy array and standardize
        X = np.array(combined_data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Generate embeddings for ward descriptions
        ward_descriptions = [
            f"Ward {info['ward_name']} from {info['dataset']} with {info['features']['total_illiterates']} illiterates"
            for info in ward_info
        ]
        
        embeddings = self.generate_embeddings(ward_descriptions)
        
        # Combine statistical features with embeddings
        if embeddings.shape[0] == X_scaled.shape[0]:
            X_combined = np.hstack([X_scaled, embeddings])
        else:
            X_combined = X_scaled
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(X_combined, max_k=min(8, len(ward_info)//2))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_combined)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(X_combined, cluster_labels) if optimal_k > 1 else 0
        
        # Organize results by clusters
        clusters = {}
        for i, (ward, label) in enumerate(zip(ward_info, cluster_labels)):
            cluster_id = f"cluster_{label}"
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    "cluster_id": label,
                    "wards": [],
                    "characteristics": {}
                }
            
            clusters[cluster_id]["wards"].append({
                "ward_name": ward['ward_name'],
                "dataset": ward['dataset'],
                "total_illiterates": ward['features']['total_illiterates'],
                "male_ratio": round(ward['features']['male_ratio'], 3),
                "female_ratio": round(ward['features']['female_ratio'], 3),
                "gender_disparity": round(ward['features']['gender_disparity'], 3)
            })
        
        # Calculate cluster characteristics
        for cluster_id, cluster_data in clusters.items():
            wards = cluster_data["wards"]
            if wards:
                avg_illiterates = np.mean([w['total_illiterates'] for w in wards])
                avg_male_ratio = np.mean([w['male_ratio'] for w in wards])
                avg_female_ratio = np.mean([w['female_ratio'] for w in wards])
                avg_disparity = np.mean([w['gender_disparity'] for w in wards])
                
                cluster_data["characteristics"] = {
                    "avg_total_illiterates": round(avg_illiterates, 1),
                    "avg_male_ratio": round(avg_male_ratio, 3),
                    "avg_female_ratio": round(avg_female_ratio, 3),
                    "avg_gender_disparity": round(avg_disparity, 3),
                    "ward_count": len(wards),
                    "datasets_represented": list(set(w['dataset'] for w in wards))
                }
        
        return {
            "clustering_results": {
                "total_wards": len(ward_info),
                "num_clusters": optimal_k,
                "silhouette_score": round(silhouette_avg, 3),
                "clusters": clusters
            },
            "methodology": {
                "algorithm": "K-Means with Hugging Face embeddings",
                "features_used": ["total_illiterates", "male_ratio", "female_ratio", "gender_disparity", "semantic_embeddings"],
                "preprocessing": "StandardScaler normalization",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    
    def detect_anomalies(self, csv_files: List[str]) -> Dict[str, Any]:
        """Detect anomalous wards using Isolation Forest and Hugging Face embeddings."""
        print("Loading and processing datasets for anomaly detection...")
        
        # Combine data from all CSV files
        combined_data = []
        ward_info = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_name = Path(csv_file).stem
                
                # Extract ward-level features
                if 'wardname' in df.columns:
                    ward_groups = df.groupby('wardname').agg({
                        'male': 'sum',
                        'female': 'sum',
                        'transgender': 'sum' if 'transgender' in df.columns else lambda x: 0
                    }).reset_index()
                    
                    for _, row in ward_groups.iterrows():
                        total = row['male'] + row['female'] + row.get('transgender', 0)
                        if total > 0:
                            features = {
                                'total_illiterates': total,
                                'male_ratio': row['male'] / total,
                                'female_ratio': row['female'] / total,
                                'transgender_ratio': row.get('transgender', 0) / total,
                                'gender_disparity': abs(row['female'] - row['male']) / total,
                                'male_count': row['male'],
                                'female_count': row['female'],
                                'dataset': file_name
                            }
                            
                            combined_data.append([
                                features['total_illiterates'],
                                features['male_ratio'],
                                features['female_ratio'],
                                features['gender_disparity'],
                                features['male_count'],
                                features['female_count']
                            ])
                            
                            ward_info.append({
                                'ward_name': row['wardname'],
                                'dataset': file_name,
                                'features': features
                            })
            
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if not combined_data:
            return {"error": "No valid data found for anomaly detection"}
        
        # Convert to numpy array and standardize
        X = np.array(combined_data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Generate embeddings for ward descriptions
        ward_descriptions = [
            f"Ward {info['ward_name']} from {info['dataset']} with {info['features']['total_illiterates']} total illiterates, {info['features']['male_count']} male and {info['features']['female_count']} female"
            for info in ward_info
        ]
        
        embeddings = self.generate_embeddings(ward_descriptions)
        
        # Combine statistical features with embeddings
        if embeddings.shape[0] == X_scaled.shape[0]:
            X_combined = np.hstack([X_scaled, embeddings])
        else:
            X_combined = X_scaled
        
        # Perform anomaly detection
        isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        anomaly_labels = isolation_forest.fit_predict(X_combined)
        anomaly_scores = isolation_forest.decision_function(X_combined)
        
        # Identify anomalies
        anomalies = []
        normal_wards = []
        
        for i, (ward, is_anomaly, score) in enumerate(zip(ward_info, anomaly_labels, anomaly_scores)):
            ward_data = {
                "ward_name": ward['ward_name'],
                "dataset": ward['dataset'],
                "total_illiterates": ward['features']['total_illiterates'],
                "male_count": ward['features']['male_count'],
                "female_count": ward['features']['female_count'],
                "male_ratio": round(ward['features']['male_ratio'], 3),
                "female_ratio": round(ward['features']['female_ratio'], 3),
                "gender_disparity": round(ward['features']['gender_disparity'], 3),
                "anomaly_score": round(float(score), 4)
            }
            
            if is_anomaly == -1:  # Anomaly detected
                # Determine anomaly type
                anomaly_type = self._classify_anomaly_type(ward['features'], combined_data)
                ward_data["anomaly_type"] = anomaly_type
                ward_data["anomaly_reason"] = self._explain_anomaly(ward['features'], anomaly_type)
                anomalies.append(ward_data)
            else:
                normal_wards.append(ward_data)
        
        # Sort anomalies by score (most anomalous first)
        anomalies.sort(key=lambda x: x['anomaly_score'])
        
        # Calculate statistics
        total_wards = len(ward_info)
        anomaly_count = len(anomalies)
        anomaly_percentage = (anomaly_count / total_wards) * 100 if total_wards > 0 else 0
        
        return {
            "anomaly_detection_results": {
                "total_wards_analyzed": total_wards,
                "anomalies_detected": anomaly_count,
                "anomaly_percentage": round(anomaly_percentage, 1),
                "anomalies": anomalies[:10],  # Top 10 most anomalous
                "summary_statistics": {
                    "avg_total_illiterates": round(np.mean([w['features']['total_illiterates'] for w in ward_info]), 1),
                    "avg_gender_disparity": round(np.mean([w['features']['gender_disparity'] for w in ward_info]), 3),
                    "datasets_analyzed": list(set(w['dataset'] for w in ward_info))
                }
            },
            "methodology": {
                "algorithm": "Isolation Forest with Hugging Face embeddings",
                "contamination_rate": 0.1,
                "features_used": ["total_illiterates", "gender_ratios", "gender_disparity", "semantic_embeddings"],
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
            }
        }
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 8) -> int:
        """Find optimal number of clusters using elbow method."""
        if len(X) < 2:
            return 1
        
        max_k = min(max_k, len(X) - 1)
        if max_k < 2:
            return 1
        
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) >= 2:
            diffs = np.diff(inertias)
            if len(diffs) >= 2:
                second_diffs = np.diff(diffs)
                elbow_idx = np.argmax(second_diffs) + 2
                return min(k_range[elbow_idx], max_k)
        
        return min(3, max_k)  # Default to 3 clusters
    
    def _classify_anomaly_type(self, features: Dict, all_data: List) -> str:
        """Classify the type of anomaly based on features."""
        total = features['total_illiterates']
        gender_disparity = features['gender_disparity']
        
        # Calculate percentiles for comparison
        all_totals = [row[0] for row in all_data]  # total_illiterates is first feature
        all_disparities = [row[3] for row in all_data]  # gender_disparity is fourth feature
        
        total_percentile = np.percentile(all_totals, 90)
        disparity_percentile = np.percentile(all_disparities, 90)
        
        if total > total_percentile:
            return "high_illiteracy"
        elif total < np.percentile(all_totals, 10):
            return "low_illiteracy"
        elif gender_disparity > disparity_percentile:
            return "high_gender_disparity"
        elif gender_disparity < np.percentile(all_disparities, 10):
            return "low_gender_disparity"
        else:
            return "statistical_outlier"
    
    def _explain_anomaly(self, features: Dict, anomaly_type: str) -> str:
        """Provide human-readable explanation for the anomaly."""
        explanations = {
            "high_illiteracy": f"Exceptionally high illiteracy count ({features['total_illiterates']}) compared to other wards",
            "low_illiteracy": f"Unusually low illiteracy count ({features['total_illiterates']}) compared to other wards",
            "high_gender_disparity": f"Significant gender disparity ({features['gender_disparity']:.3f}) in illiteracy rates",
            "low_gender_disparity": f"Unusually balanced gender distribution ({features['gender_disparity']:.3f}) in illiteracy",
            "statistical_outlier": "Statistical pattern differs significantly from typical wards"
        }
        return explanations.get(anomaly_type, "Anomalous pattern detected in illiteracy data")
