# CrewAI ML Integration

## Overview

The RTGS CLI now features a powerful CrewAI-powered ML agent that provides advanced clustering and anomaly detection capabilities. This hybrid AI system combines the reasoning power of Groq with the multi-agent orchestration of CrewAI for sophisticated data analysis.

## Architecture

### Hybrid AI System
- **Groq API**: Real-time insights and policy summaries
- **CrewAI**: Multi-agent ML orchestration for clustering and anomaly detection
- **Hugging Face**: Optional embeddings and semantic analysis
- **MCP Server**: Unified access for external AI agents

### CrewAI Agents

#### 1. Data Analyst Agent
**Role**: Data Analyst  
**Goal**: Analyze ward-level illiteracy data and extract meaningful patterns  
**Expertise**: Governance and demographic data analysis, pattern identification

#### 2. Clustering Specialist Agent
**Role**: Clustering Specialist  
**Goal**: Group wards with similar illiteracy patterns using advanced clustering techniques  
**Expertise**: K-means clustering, unsupervised learning, ward grouping

#### 3. Anomaly Detection Specialist Agent
**Role**: Anomaly Detection Specialist  
**Goal**: Identify unusual patterns and outliers in ward-level illiteracy data  
**Expertise**: Isolation Forest, statistical outlier identification, anomaly classification

## Features

### Ward Clustering
- **Multi-dataset Analysis**: Process multiple CSV files simultaneously
- **Feature Engineering**: Automatic extraction of illiteracy patterns, gender ratios, and disparities
- **Optimal Clustering**: Automatic determination of optimal cluster count using silhouette analysis
- **Semantic Understanding**: Optional Hugging Face embeddings for contextual clustering
- **Governance Insights**: Cluster characteristics tailored for policy decision-making

### Anomaly Detection
- **Statistical Outliers**: Isolation Forest algorithm for robust anomaly detection
- **Anomaly Classification**: Automatic categorization of anomaly types:
  - High illiteracy outliers
  - Low illiteracy outliers
  - Gender disparity anomalies
  - Statistical outliers
- **Explainable AI**: Human-readable explanations for each detected anomaly
- **Prioritization**: Anomalies ranked by severity score

## CLI Commands

### Clustering Command
```bash
# Cluster wards across multiple datasets
python cli.py cluster data/cleaned/*.csv
python cli.py cluster data/cleaned/dataset1.csv data/cleaned/dataset2.csv

# Example output locations
outputs/clusters.json    # Detailed clustering results
```

### Anomaly Detection Command
```bash
# Detect anomalies across multiple datasets
python cli.py anomalies data/cleaned/*.csv
python cli.py anomalies data/cleaned/dataset1.csv data/cleaned/dataset2.csv

# Example output locations
outputs/anomalies.json   # Detailed anomaly results
```

## MCP Integration

### New MCP Tools

#### `rtgs_cluster`
**Description**: Cluster wards across multiple CSV files using CrewAI ML  
**Parameters**:
- `csv_files` (array): List of CSV file paths for clustering analysis

**Example**:
```python
result = await client.call_tool("rtgs_cluster", {
    "csv_files": ["data/cleaned/dataset1.csv", "data/cleaned/dataset2.csv"]
})
```

#### `rtgs_anomalies`
**Description**: Detect anomalous wards using CrewAI ML  
**Parameters**:
- `csv_files` (array): List of CSV file paths for anomaly detection

**Example**:
```python
result = await client.call_tool("rtgs_anomalies", {
    "csv_files": ["data/cleaned/dataset1.csv", "data/cleaned/dataset2.csv"]
})
```

## Output Formats

### Clustering Results (`outputs/clusters.json`)
```json
{
  "clustering_results": {
    "total_wards": 45,
    "num_clusters": 4,
    "silhouette_score": 0.623,
    "clusters": {
      "cluster_0": {
        "cluster_id": 0,
        "wards": [
          {
            "ward_name": "Revenue Ward No 1",
            "dataset": "dataset_transformed",
            "total_illiterates": 1250,
            "male_ratio": 0.445,
            "female_ratio": 0.555,
            "gender_disparity": 0.110
          }
        ],
        "characteristics": {
          "avg_total_illiterates": 1180.5,
          "avg_male_ratio": 0.448,
          "avg_female_ratio": 0.552,
          "avg_gender_disparity": 0.104,
          "ward_count": 12,
          "datasets_represented": ["dataset_transformed"]
        }
      }
    }
  },
  "methodology": {
    "algorithm": "K-Means with CrewAI orchestration",
    "features_used": ["total_illiterates", "male_ratio", "female_ratio", "gender_disparity"],
    "preprocessing": "StandardScaler normalization",
    "ai_framework": "CrewAI multi-agent system"
  },
  "crew_insights": "CrewAI agent analysis and recommendations"
}
```

### Anomaly Detection Results (`outputs/anomalies.json`)
```json
{
  "anomaly_detection_results": {
    "total_wards_analyzed": 45,
    "anomalies_detected": 4,
    "anomaly_percentage": 8.9,
    "anomalies": [
      {
        "ward_name": "Nadergul",
        "dataset": "dataset_transformed",
        "total_illiterates": 3245,
        "male_count": 1456,
        "female_count": 1789,
        "male_ratio": 0.449,
        "female_ratio": 0.551,
        "gender_disparity": 0.102,
        "anomaly_score": -0.2341,
        "anomaly_type": "high_illiteracy",
        "anomaly_reason": "Exceptionally high illiteracy count (3245) compared to other wards"
      }
    ],
    "summary_statistics": {
      "avg_total_illiterates": 1245.6,
      "avg_gender_disparity": 0.089,
      "datasets_analyzed": ["dataset_transformed"]
    }
  },
  "methodology": {
    "algorithm": "Isolation Forest with CrewAI orchestration",
    "contamination_rate": 0.1,
    "features_used": ["total_illiterates", "gender_ratios", "gender_disparity"],
    "ai_framework": "CrewAI multi-agent system"
  },
  "crew_insights": "CrewAI agent analysis and recommendations"
}
```

## Technical Implementation

### CrewAI Tools
The ML agent uses specialized CrewAI tools:

- **`analyze_ward_data`**: Extract features from multiple CSV files
- **`perform_clustering_analysis`**: Execute K-means clustering with optimal parameters
- **`perform_anomaly_detection`**: Run Isolation Forest anomaly detection

### Fallback Mechanisms
- **CrewAI Unavailable**: Automatic fallback to direct scikit-learn implementation
- **API Failures**: Graceful degradation with error handling
- **Missing Dependencies**: Clear error messages and alternative approaches

### Performance Optimizations
- **Batch Processing**: Efficient handling of multiple CSV files
- **Memory Management**: Optimized for large datasets
- **Async Operations**: Non-blocking execution in MCP server
- **Caching**: Intelligent caching of processed results

## Use Cases

### Governance Applications

#### 1. Resource Allocation
- **Cluster Analysis**: Identify ward groups with similar needs for targeted resource allocation
- **Anomaly Detection**: Flag wards requiring immediate attention or investigation

#### 2. Policy Development
- **Pattern Recognition**: Understand regional patterns in illiteracy for policy design
- **Outlier Analysis**: Identify successful or problematic wards for case studies

#### 3. Monitoring & Evaluation
- **Trend Analysis**: Track changes in ward clustering over time
- **Performance Monitoring**: Detect unusual changes in ward performance

### Research Applications

#### 1. Demographic Studies
- **Similarity Analysis**: Group wards by demographic characteristics
- **Outlier Studies**: Investigate unusual demographic patterns

#### 2. Intervention Planning
- **Target Identification**: Use clustering to identify intervention targets
- **Impact Assessment**: Monitor anomalies before/after interventions

## Integration Examples

### Workflow Automation
```python
# Automated analysis pipeline
async def analyze_governance_data():
    # Step 1: Process data
    await mcp_client.call_tool("rtgs_pipeline", {"file_path": "raw_data.csv"})
    
    # Step 2: Cluster analysis
    cluster_result = await mcp_client.call_tool("rtgs_cluster", {
        "csv_files": ["data/cleaned/dataset_transformed.csv"]
    })
    
    # Step 3: Anomaly detection
    anomaly_result = await mcp_client.call_tool("rtgs_anomalies", {
        "csv_files": ["data/cleaned/dataset_transformed.csv"]
    })
    
    # Step 4: Generate insights
    insights = await mcp_client.call_tool("rtgs_insights", {
        "file_path": "data/cleaned/dataset_transformed.csv"
    })
    
    return {
        "clusters": cluster_result,
        "anomalies": anomaly_result,
        "insights": insights
    }
```

### Dashboard Integration
```python
# Real-time dashboard updates
def update_governance_dashboard():
    # Get latest clustering results
    with open("outputs/clusters.json") as f:
        clusters = json.load(f)
    
    # Get anomalies
    with open("outputs/anomalies.json") as f:
        anomalies = json.load(f)
    
    # Update dashboard widgets
    update_cluster_visualization(clusters)
    update_anomaly_alerts(anomalies)
    update_ward_recommendations(clusters, anomalies)
```

## Best Practices

### Data Preparation
1. **Consistent Schema**: Ensure all CSV files have consistent column names
2. **Data Quality**: Clean data before clustering/anomaly detection
3. **Feature Selection**: Include relevant demographic and geographic features

### Analysis Configuration
1. **Cluster Count**: Let the algorithm determine optimal clusters for governance data
2. **Anomaly Threshold**: 10% contamination rate works well for governance datasets
3. **Feature Scaling**: Always use StandardScaler for consistent results

### Result Interpretation
1. **Cluster Characteristics**: Focus on governance-relevant cluster features
2. **Anomaly Prioritization**: Address high-score anomalies first
3. **Temporal Analysis**: Compare results across different time periods

## Troubleshooting

### Common Issues

1. **CrewAI Import Error**
   - Install crewai: `pip install crewai`
   - Falls back to direct ML implementation

2. **Memory Issues with Large Datasets**
   - Process files in smaller batches
   - Use data sampling for initial analysis

3. **Poor Clustering Results**
   - Check data quality and feature relevance
   - Adjust preprocessing parameters
   - Consider additional features

4. **Too Many/Few Anomalies**
   - Adjust contamination rate (default: 0.1)
   - Review feature engineering
   - Check data distribution

### Performance Tuning

1. **Clustering Performance**
   - Limit maximum clusters to 8
   - Use silhouette score for validation
   - Consider PCA for high-dimensional data

2. **Anomaly Detection Performance**
   - Tune contamination rate based on domain knowledge
   - Use ensemble methods for better accuracy
   - Validate results with domain experts

## Future Enhancements

### Planned Features
1. **Time Series Clustering**: Cluster wards based on temporal patterns
2. **Hierarchical Clustering**: Multi-level ward grouping
3. **Deep Learning Integration**: Neural network-based anomaly detection
4. **Interactive Visualization**: Web-based cluster and anomaly exploration
5. **Automated Reporting**: AI-generated governance reports from ML results

### Integration Roadmap
1. **Advanced MCP Tools**: More sophisticated ML operations
2. **Real-time Processing**: Streaming data analysis
3. **Multi-modal Analysis**: Combine text, numeric, and geographic data
4. **Federated Learning**: Privacy-preserving multi-dataset analysis
