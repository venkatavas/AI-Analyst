## 🎬 Demo Video

[🎥 Watch 2-minute demo](https://drive.google.com/file/d/1EKN5gNk23aHWbroDuWQn9n8I7mXdUm01/view?usp=drivesdk) - See the complete pipeline in action


# 🏛️ RTGS CLI - Real-Time Governance System

[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)](https://github.com/your-repo)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![AI Powered](https://img.shields.io/badge/AI-Groq%20%2B%20Local%20ML-orange)](https://groq.com)
[![ML Analytics](https://img.shields.io/badge/ML-Clustering%20%2B%20Anomaly%20Detection-purple)](https://github.com/your-repo)

## 🎯 Overview

The **Real-Time Governance System (RTGS) CLI** is a production-ready **AI-powered governance data analysis platform** that transforms raw administrative data into actionable policy insights. Built with enterprise-grade stability and comprehensive error handling, it combines Groq LLM integration with local ML analytics for complete governance workflow automation.

**🎉 Multi-Dataset Pipeline**: Processes 5 datasets simultaneously with automatic backup system preserving all results.

## 🚀 Key Achievements

- ✅ **Production-Ready Stability**: Zero hanging imports, robust error handling
- ✅ **AI-Powered Analytics**: Groq LLM + Local ML statistical analysis  
- ✅ **Complete ML Pipeline**: Clustering, anomaly detection, statistical insights
- ✅ **Enterprise-Grade**: Processes 687+ records with 630 ward-level analytics
- ✅ **Professional Output**: Multiple report formats (MD, JSON, TXT, HTML)

## 🤖 **AI Architecture**

### **AI Integration Architecture**
- **Groq LLM (llama-3.3-70b-versatile)**: Narrative policy insights and storytelling
- **Local Mathematical Analysis**: Pure Python statistical processing
- **SimpleMlAgent**: Local ML clustering and anomaly detection
- **Smart Fallbacks**: Graceful degradation when APIs unavailable

### **Specialized Agents**
- **IngestionAgent**: CSV standardization and schema validation
- **CleaningAgent**: Data quality assessment and duplicate removal  
- **TransformationAgent**: Feature engineering and statistical aggregations
- **InsightsAgent**: AI-powered policy recommendations
- **SimpleMlAgent**: ML clustering and outlier detection

## 📊 **Analytics Capabilities**

### **Statistical Analysis**
- **Ward-Level Insights**: 630+ wards analyzed with comprehensive statistics
- **Outlier Detection**: IQR-based anomaly identification with deviation metrics
- **Gender Analysis**: Female illiteracy ratio tracking and high-risk ward identification
- **Clustering Analysis**: 5-cluster ML segmentation with ward groupings

### **AI-Powered Insights** 
- **Narrative Generation**: Human-readable policy stories from data patterns (Groq LLM)
- **Mathematical Analysis**: Statistical coefficients, variance analysis, clustering metrics (Local)
- **Policy Recommendations**: Actionable insights for governance decision-making
- **Trend Analysis**: Historical patterns and predictive indicators

## ⚡ **Core Features**

### 🔄 **Complete Data Pipeline**
```
Raw CSV → Ingest → Clean → Transform → AI Insights → Reports
```
- **Schema Validation**: Automatic CSV structure standardization
- **Quality Assessment**: Missing data detection and duplicate removal
- **Feature Engineering**: Statistical aggregations and derived metrics
- **AI Enhancement**: Dual-API analysis with smart fallbacks

### 🤖 **Production-Ready AI Integration**
- **Environment-Controlled Loading**: `ENABLE_PLOTLY=true` for interactive charts
- **Robust Error Handling**: Graceful fallbacks when APIs unavailable
- **Smart Import Management**: No hanging imports, timeout protection
- **Multiple Output Formats**: HTML, PNG, TXT, JSON, MD, CSV

### 📈 **Visualization System**
- **Interactive Charts**: Plotly-based ward analysis (optional)
- **Text-Based Fallbacks**: ASCII charts for universal compatibility
- **Professional Reports**: Markdown with embedded statistics
- **Export Options**: Multiple formats for different use cases

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone repository
git clone <your-repository-url>
cd rtgs-cli

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### **Environment Setup**
```bash
# Create .env file with API keys
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Optional: Enable interactive visualizations
ENABLE_PLOTLY=true
```

### **Usage Examples**

#### **🚀 Multi-Dataset Pipeline (Recommended)**
```bash
# Windows
.\run_demo.bat

# Cross-platform Python
python run_demo.py
```
**Processes all 5 datasets automatically:**
- **Illiterate_Khammam_Rural.csv** (687 records, 630+ wards)
- **Illiterate_Nalgonda_Rural.csv** (governance data)
- **Illiterate_Rangareddy_Urban_Area.csv** (urban governance)
- **Illiterate_Rangareddy_Urban_Area1.csv** (urban governance)
- **Skill Development.csv** (activity tracking)

**Each dataset gets complete 6-step analysis:**
1. Data ingestion and standardization
2. Data cleaning and quality assessment
3. Statistical transformation and feature engineering
4. AI insights generation (Groq LLM + Local ML)
5. ML clustering analysis
6. Anomaly detection (IQR-based outliers)

**Results automatically backed up to dataset-specific folders!**

#### **Individual Commands** (if needed)
```bash
# Process governance data step-by-step
python cli.py ingest data/raw/Illiterate_Khammam_Rural.csv
python cli.py clean data/raw/Illiterate_Khammam_Rural_standardized.csv  
python cli.py transform data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned.csv
python cli.py insights data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
python cli.py cluster data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
python cli.py anomalies data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
```

#### **Get Help**
```bash
python cli.py --help
python cli.py insights --help
```

## 📋 Multi-Dataset Results

### **Production Analytics Delivered:**
- **5 Complete Datasets Processed**: All with full 6-step analysis pipeline
- **Dataset-Specific Results**: Automatic backup system preserves all outputs
- **Groq AI Insights**: Real-time policy recommendations for each region
- **ML Analytics**: Ward clustering and anomaly detection across all datasets
- **Processing Speed**: Complete multi-dataset pipeline in under 5 minutes
- **Output Quality**: Professional governance reports ready for policy decisions

### **Output Structure:**
```
outputs/
├── Illiterate_Khammam_Rural_results/          # Complete analytics
├── Illiterate_Nalgonda_Rural_results/         # Complete analytics
├── Illiterate_Rangareddy_Urban_Area_results/  # Complete analytics
├── Illiterate_Rangareddy_Urban_Area1_results/ # Complete analytics
└── Skill Development_results/                 # Activity analytics
```

**Each folder contains:** Schema analysis, cleaning reports, transformations, AI insights, ML clusters, anomaly detection, and visualizations.

## 📚 Documentation

- **[Architecture Overview](docs/README_architecture.md)** - Technical design and agent workflow
- **[Usage Guide](docs/README_usage.md)** - Step-by-step commands and examples  
- **[Output Reference](docs/README_outputs.md)** - Understanding generated reports
- **[Dataset Information](docs/README_dataset.md)** - Data source and methodology
- **[CrewAI ML Integration](docs/README_crewai.md)** - Multi-agent ML capabilities
- **[MCP Server Guide](docs/README_mcp.md)** - Model Context Protocol integration

## 🛠️ Installation

```bash
git clone <repository-url>
cd rtgs-cli
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## 📊 Multi-Dataset Manifest

### **Governance Datasets (4 datasets)**

#### **1. Illiterate_Khammam_Rural.csv**
- **Records**: 687 administrative entries
- **Wards**: 630+ unique administrative units
- **Geographic Scope**: Rural Khammam district, Telangana
- **Anomalies**: 29 high-risk wards identified

#### **2. Illiterate_Nalgonda_Rural.csv**
- **Geographic Scope**: Rural Nalgonda district, Telangana
- **Analysis**: Complete ward-level governance analytics

#### **3. Illiterate_Rangareddy_Urban_Area.csv**
- **Geographic Scope**: Urban Rangareddy district, Telangana
- **Analysis**: Urban governance pattern analysis

#### **4. Illiterate_Rangareddy_Urban_Area1.csv**
- **Geographic Scope**: Urban Rangareddy district (Area 1), Telangana
- **Analysis**: Detailed urban sub-area analytics

### **Non-Governance Dataset**

#### **5. Skill Development.csv**
- **Records**: 4 activity tracking entries
- **Scope**: Activity monitoring over 2015-17 period
- **Analysis**: Statistical summary and trend analysis

### **Common Data Schema (Governance)**
- `district`: Administrative district name
- `mandal`: Sub-district administrative unit  
- `wardname`: Ward-level governance unit (primary analysis key)
- `male`: Male illiteracy count
- `female`: Female illiteracy count
- `transgender`: Transgender illiteracy count
- `total_illiterates`: Computed total illiteracy per ward

### **Comprehensive Analytics Results**
- **Total Records Processed**: 660,495+ illiterates across all datasets
- **Ward Coverage**: 630+ administrative units analyzed
- **Anomaly Detection**: 29 critical wards requiring immediate attention
- **Gender Analysis**: 59.7% female illiteracy rate identified
- **Risk Categorization**: Critical, High, Medium levels assigned
- **Policy Insights**: AI-powered recommendations for all regions


## 🏗️ Built With

- **Python 3.11+** - Core runtime
- **Pandas** - Data processing engine  
- **CrewAI** - Multi-agent orchestration framework
- **Matplotlib** - Visualization generation
- **Groq API** - AI-powered narrative insights (configurable)

---

*Developed for hackathons, recruitment demos, and real-world governance applications*
