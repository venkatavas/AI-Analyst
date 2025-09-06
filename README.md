# RTGS CLI - Real-Time Governance System

## Overview

The Real-Time Governance System (RTGS) CLI is a cutting-edge **Hybrid AI System** that combines multiple AI technologies for comprehensive governance data analysis. It features a multi-agent architecture powered by **Groq**, **CrewAI**, and **Hugging Face** APIs, with **MCP server integration** for seamless AI interoperability.

## Hybrid AI Architecture

### 🤖 **Multi-AI Integration**
- **Groq API**: Real-time policy insights and governance summaries
- **CrewAI**: Multi-agent ML orchestration for clustering and anomaly detection  
- **Hugging Face**: Semantic embeddings and advanced NLP capabilities
- **MCP Server**: Model Context Protocol for external AI agent access

### 🎯 **Specialized AI Agents**
- **Data Analyst Agent**: Governance data pattern recognition
- **Clustering Specialist**: Ward grouping using K-means with AI orchestration
- **Anomaly Detection Specialist**: Statistical outlier identification with explanations
- **Insights Agent**: Policy recommendation generation

## Problem Statement

Governance data analysis faces critical challenges:
- **Fragmented datasets** across multiple sources and formats
- **Inconsistent data quality** affecting policy decisions
- **Manual analysis bottlenecks** limiting real-time insights
- **Lack of AI integration** for advanced pattern recognition
- **No standardized ML workflows** for governance applications

## Solution

RTGS CLI delivers a comprehensive **Hybrid AI Solution**:
- **Automated data pipeline** with multi-agent processing
- **Real-time AI insights** using state-of-the-art language models
- **Advanced ML capabilities** through CrewAI orchestration
- **Professional governance reporting** with actionable recommendations
- **API-first architecture** enabling seamless integration

## ⚡ Key Features

### 🔄 **Multi-Agent Data Pipeline**
- **Ingestion Agent**: CSV standardization and schema validation
- **Cleaning Agent**: Data quality assessment and duplicate removal
- **Transformation Agent**: Statistical aggregations and feature engineering
- **Insights Agent**: Policy recommendations with AI enhancement

### 🤖 **Advanced AI Capabilities**
- **Groq Integration**: Real-time policy insights and governance summaries
- **CrewAI ML**: Multi-agent clustering and anomaly detection
- **Hugging Face**: Semantic embeddings and NLP processing
- **Hybrid AI Orchestration**: Coordinated multi-AI system responses

### 🔗 **Enterprise Integration**
- **MCP Server**: Model Context Protocol for AI agent interoperability
- **API-First Design**: RESTful access to all pipeline functions
- **CLI Interface**: Direct command-line usage for data scientists
- **Async Processing**: Non-blocking operations for large datasets

### 📊 **Professional Outputs**
- **Governance Reports**: Policy-focused insights and recommendations
- **Statistical Analysis**: Ward-level clustering and anomaly detection
- **Data Visualizations**: Automated charts and graphs
- **JSON/CSV Exports**: Machine-readable results for integration

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys in .env file
GROQ_API_KEY=your_groq_key_here
HUGGINGFACE_API_KEY=your_hf_key_here
```

### Complete Pipeline Demo
```bash
# Run full demonstration
python demo_full_pipeline.py

# Or run individual commands
python cli.py ingest data/raw/your_data.csv
python cli.py clean data/cleaned/your_data_standardized.csv
python cli.py transform data/cleaned/your_data_cleaned.csv
python cli.py insights data/cleaned/your_data_transformed.csv

# New CrewAI ML commands
python cli.py cluster data/cleaned/*.csv
python cli.py anomalies data/cleaned/*.csv
```

### MCP Server Integration
```bash
# Start MCP server for AI agent access
python mcp_server.py

# Available MCP tools: rtgs_ingest, rtgs_clean, rtgs_transform, 
# rtgs_insights, rtgs_cluster, rtgs_anomalies, rtgs_pipeline
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
