# 🏗️ Production Architecture Overview

## 🚀 **Production-Ready Multi-Agent Pipeline**

RTGS CLI implements a **production-grade, dual-API architecture** that processes governance data through specialized agents with enterprise-level stability:

```
Raw CSV Data (687+ records)
     ↓
┌─────────────────┐
│ Ingestion Agent │ → Schema validation, standardization
└─────────────────┘
     ↓
┌─────────────────┐
│ Cleaning Agent  │ → Quality assessment, duplicate removal
└─────────────────┘
     ↓
┌─────────────────┐
│Transform Agent  │ → Feature engineering, ward aggregation
└─────────────────┘
     ↓
┌─────────────────┐
│ Insights Agent  │ → Dual-API AI analysis (Groq + HuggingFace)
└─────────────────┘
     ↓
┌─────────────────┐
│ SimpleMlAgent   │ → ML clustering, anomaly detection
└─────────────────┘
     ↓
Professional Reports & Visualizations
```

## 🤖 **Dual-API AI Integration**

### **Production AI Architecture**
```
Input Data → Groq LLM (Narrative) + HuggingFace (Mathematical) → Combined Insights
                ↓                           ↓
        Policy Stories              Statistical Analysis
                ↓                           ↓
                    Combined AI Report
```

## Agent Responsibilities

### 1. Ingestion Agent (`agents/ingestion_agent.py`)
- **Purpose**: Data intake and initial validation
- **Functions**:
  - Column name standardization (lowercase, underscores)
  - Schema generation (data types, null counts, unique values)
  - CSV export with standardized format
- **Outputs**: 
  - `*_standardized.csv`
  - `schema_summary.json`
  - `schema_summary.csv`

### 2. Cleaning Agent (`agents/cleaning_agent.py`)
- **Purpose**: Data quality improvement
- **Functions**:
  - Duplicate row removal
  - Missing value imputation (mean for numeric, "Unknown" for categorical)
  - Outlier detection and replacement (3-sigma rule)
- **Outputs**:
  - `*_cleaned.csv`
  - `cleaning_report.json` (detailed change log)

### 3. Transformation Agent (`agents/transformation_agent.py`)
- **Purpose**: Feature engineering and aggregation
- **Functions**:
  - Ward-level illiteracy aggregation
  - Gender-based statistical breakdowns
  - Percentage calculations and ranking
  - New derived columns creation
- **Outputs**:
  - `*_transformed.csv`
  - `transformation_report.json`

### 4. Insights Agent (`agents/insights_agent.py`)
- **Purpose**: Dual-API AI analysis and policy reporting
- **Functions**:
  - **Groq LLM Integration**: Narrative policy insights using llama3-70b-8192
  - **HuggingFace API**: Mathematical analysis and statistical processing
  - **Combined AI Reports**: Merged narrative + mathematical insights
  - **Smart Fallbacks**: Local analysis when APIs unavailable
- **Outputs**:
  - `insights_report.md` (comprehensive AI analysis)
  - Console summary with key findings

### 5. SimpleMlAgent (`agents/simple_ml_agent.py`)
- **Purpose**: Local ML analytics without external dependencies
- **Functions**:
  - **K-means Clustering**: Ward segmentation (5 clusters)
  - **Anomaly Detection**: IQR-based outlier identification
  - **Statistical Analysis**: Coefficient of variation, gender ratios
  - **Pure Python Implementation**: No scikit-learn dependency
- **Outputs**:
  - `clusters.json` (cluster analysis results)
  - `anomalies.json` (outlier detection results)

## Design Principles

### Modularity
Each agent operates independently with clear input/output contracts, enabling:
- **Parallel development** of different pipeline stages
- **Easy testing** of individual components
- **Flexible deployment** (agents can be run separately or in sequence)

### Scalability
The architecture supports:
- **Multiple datasets** by adding new agent instances
- **Custom transformations** through agent subclassing
- **External integrations** via standardized interfaces

### Observability
Built-in monitoring includes:
- **Progress tracking** with visual indicators (`[→]`, `[✓]`, `[✗]`)
- **Detailed logging** of all operations
- **Comprehensive reporting** at each stage
- **Error handling** with graceful degradation

## Technology Stack

### Core Framework
- **Python 3.11+**: Modern language features and performance
- **Pandas**: High-performance data manipulation
- **Argparse**: CLI argument parsing and validation

### AI/ML Integration
- **Groq API**: LLM integration (llama3-70b-8192) for narrative insights
- **HuggingFace API**: Mathematical analysis and statistical processing
- **SimpleMlAgent**: Local ML clustering and anomaly detection
- **Plotly**: Interactive visualizations (environment-controlled)
- **Smart Fallbacks**: Text-based charts and local analysis

### Data Processing
- **JSON**: Structured reporting format
- **CSV**: Standardized data exchange
- **Markdown**: Human-readable documentation

## Future Enhancements

### 1. Real-Time Processing
- **Stream processing** for live data feeds
- **Event-driven architecture** with message queues
- **Incremental updates** for large datasets

### 2. Multi-Dataset Joins
- **Cross-dataset correlation** analysis
- **Temporal trend analysis** across multiple years
- **Geographic data integration** with mapping

### 3. Advanced AI Features
- **Predictive modeling** for policy impact forecasting
- **Natural language queries** for ad-hoc analysis
- **Automated anomaly detection** in governance metrics

### 4. Production Deployment
- **Docker containerization** for consistent deployment
- **REST API** for web application integration
- **Database connectivity** for enterprise data sources
- **Role-based access control** for sensitive data

## Performance Characteristics

### Processing Speed
- **Small datasets** (< 10K rows): < 10 seconds end-to-end
- **Medium datasets** (10K-100K rows): < 60 seconds end-to-end
- **Large datasets** (100K+ rows): Scales linearly with row count

### Memory Usage
- **Efficient pandas operations** minimize memory footprint
- **Streaming processing** for datasets larger than available RAM
- **Garbage collection** between pipeline stages

### Error Recovery
- **Graceful degradation** when optional features fail
- **Detailed error reporting** with actionable suggestions
- **Partial pipeline execution** for debugging and development
