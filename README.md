# 🏛️ RTGS CLI - Real-Time Governance System

[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)](https://github.com/your-repo)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![AI Powered](https://img.shields.io/badge/AI-Groq%20%2B%20HuggingFace-orange)](https://groq.com)
[![ML Analytics](https://img.shields.io/badge/ML-Clustering%20%2B%20Anomaly%20Detection-purple)](https://github.com/your-repo)

## 🎯 Overview

The **Real-Time Governance System (RTGS) CLI** is a production-ready **AI-powered governance data analysis platform** that transforms raw administrative data into actionable policy insights. Built with enterprise-grade stability and comprehensive error handling, it combines dual-API AI integration with advanced ML analytics for complete governance workflow automation.

## 🚀 Key Achievements

- ✅ **Production-Ready Stability**: Zero hanging imports, robust error handling
- ✅ **Dual-API AI Integration**: Groq LLM + HuggingFace mathematical analysis  
- ✅ **Complete ML Pipeline**: Clustering, anomaly detection, statistical insights
- ✅ **Enterprise-Grade**: Processes 687+ records with 630 ward-level analytics
- ✅ **Professional Output**: Multiple report formats (MD, JSON, TXT, HTML)

## 🤖 **AI Architecture**

### **Dual-API Integration**
- **Groq LLM (llama3-70b-8192)**: Narrative policy insights and storytelling
- **HuggingFace API**: Mathematical analysis and statistical processing
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
- **Narrative Generation**: Human-readable policy stories from data patterns
- **Mathematical Analysis**: Statistical coefficients, variance analysis, clustering metrics
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

#### **🚀 One-Command Complete Pipeline**
```bash
# Windows
.\run_demo.bat

# Cross-platform Python
python run_demo.py
```
**Executes all 6 steps automatically:**
1. Data ingestion (687 records)
2. Data cleaning and quality assessment
3. Statistical transformation and feature engineering
4. Dual-API AI insights (Groq + HuggingFace)
5. ML clustering analysis (5 clusters)
6. Anomaly detection (IQR-based outliers)

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

## 📋 Demo Results

Latest hybrid AI analysis capabilities:
- **Groq AI Insights**: Real-time policy recommendations and governance summaries
- **CrewAI ML**: Ward clustering and anomaly detection with multi-agent orchestration
- **Processing Speed**: Complete pipeline + ML analysis in under 60 seconds
- **Output Quality**: Professional governance reports with actionable insights

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

## 🎬 Demo Video

[🎥 Watch 2-minute demo](link-to-demo-video) - See the complete pipeline in action

## 🏗️ Built With

- **Python 3.11+** - Core runtime
- **Pandas** - Data processing engine  
- **CrewAI** - Multi-agent orchestration framework
- **Matplotlib** - Visualization generation
- **Groq API** - AI-powered insights (configurable)

---

*Developed for hackathons, recruitment demos, and real-world governance applications*
