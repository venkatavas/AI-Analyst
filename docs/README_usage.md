# Usage Guide

## Quick Start

### 🚀 One-Command Complete Pipeline
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

**Expected Complete Output:**
```
========================================
🚀 RTGS CLI - Complete Pipeline Demo
========================================
Production-Ready Governance Analytics
Dual-API AI Integration + ML Analytics
========================================

[→] Step 1: Ingest raw dataset (687 records)
[✓] Step 1: Ingest raw dataset (687 records) completed successfully

[→] Step 2: Clean standardized dataset
[✓] Step 2: Clean standardized dataset completed successfully

[→] Step 3: Transform cleaned dataset
[✓] Step 3: Transform cleaned dataset completed successfully

[→] Step 4: Generate dual-API insights (Groq + HuggingFace)
[✓] Step 4: Generate dual-API insights (Groq + HuggingFace) completed successfully

[→] Step 5: ML clustering analysis (SimpleMlAgent)
[✓] Step 5: ML clustering analysis (SimpleMlAgent) completed successfully

[→] Step 6: Anomaly detection (IQR-based outliers)
[✓] Step 6: Anomaly detection (IQR-based outliers) completed successfully

========================================
[✓] Complete Pipeline Demo Finished!
========================================
📊 Analytics Results Generated:
   📄 Schema Analysis: outputs/schema_summary.json
   🧹 Data Quality: outputs/cleaning_report.json
   🔄 Transformations: outputs/transformation_report.json
   🤖 Dual-API Insights: outputs/insights_report.md
   🎯 ML Clusters: outputs/clusters.json
   🚨 Anomalies: outputs/anomalies.json
   📈 Visualizations: outputs/trend.png

🎉 Production-ready governance analytics complete!
📊 687 records processed across 630+ wards
🤖 Dual-API AI analysis (Groq + HuggingFace)
🎯 5 clusters identified, 3 outliers detected
```

## Step-by-Step Usage

### 1. Ingest Command
Process raw CSV data and generate schema information.

```bash
python cli.py ingest data/raw/Illiterate_Khammam_Rural.csv
```

**Expected Output:**
```
[→] INGEST: Starting data ingestion process
[→] LOAD: Reading CSV file: Illiterate_Khammam_Rural.csv
[✓] Loaded 687 rows and 7 columns
[→] STANDARDIZE: Converting column names to lowercase with underscores
[✓] Column names standardized
[→] ANALYZE: Generating schema summary

📊 Dataset Overview:
   • Total Rows: 687
   • Total Columns: 7

📋 Column Details:
   • district (object): 687 non-null (0.0% missing), 1 unique
   • mandal (object): 687 non-null (0.0% missing), 46 unique
   • muncipality (object): 687 non-null (0.0% missing), 46 unique
   • wardname (object): 687 non-null (0.0% missing), 630 unique
   • female (int64): 687 non-null (0.0% missing), 312 unique
   • male (int64): 687 non-null (0.0% missing), 298 unique
   • transgender (int64): 687 non-null (0.0% missing), 4 unique

[→] SAVE: Saving processed files
[✓] Successfully processed Illiterate_Khammam_Rural.csv
   📁 Standardized data: data\raw\Illiterate_Khammam_Rural_standardized.csv
   📄 Schema JSON: outputs\schema_summary.json
   📊 Schema CSV: outputs\schema_summary.csv
```

### 2. Clean Command
Remove duplicates, handle missing values, and detect outliers.

```bash
python cli.py clean data/raw/Illiterate_Khammam_Rural_standardized.csv
```

**Expected Output:**
```
[→] CLEAN: Starting data cleaning process
[→] LOAD: Reading standardized file: Illiterate_Rangareddy_Urban_Area_standardized.csv
[✓] Loaded 126 rows for cleaning
[→] PROCESS: Applying data cleaning operations
[✓] Cleaning completed - 126 rows remaining

🧹 Cleaning Summary:
   • Duplicates removed: 0
   • Missing values filled: 0
   • Outliers replaced: 0

[→] SAVE: Saving cleaned data and report
[✓] Successfully cleaned Illiterate_Khammam_Rural_standardized.csv
   📁 Cleaned data: data\cleaned\Illiterate_Khammam_Rural_standardized_cleaned.csv
   📄 Cleaning report: outputs\cleaning_report.json
```

### 3. Transform Command
Aggregate data by ward and calculate derived metrics.

```bash
python cli.py transform data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned.csv
```

**Expected Output:**
```
[→] TRANSFORM: Starting data transformation process
[→] LOAD: Reading cleaned file: Illiterate_Khammam_Rural_standardized_cleaned.csv
[✓] Loaded 687 rows for transformation
[→] PROCESS: Applying transformations and aggregations
[✓] Transformations completed

🔄 Transformation Summary:
   • Total illiterates: 6,789
   • Male illiterates: 2,567
   • Female illiterates: 4,222
   • Top ward: Nadergul (456 illiterates)

[→] SAVE: Saving transformed data and generating visualization
[✓] Successfully transformed Illiterate_Khammam_Rural_standardized_cleaned.csv
   📁 Transformed data: data\cleaned\Illiterate_Khammam_Rural_cleaned_transformed.csv
   📄 Transformation report: outputs\transformation_report.json
   📊 Visualization: outputs\trend.png
```

### 4. Insights Command
Generate dual-API AI-powered policy recommendations and visualizations.

```bash
python cli.py insights data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
```

**Expected Output:**
```
[→] INSIGHTS: Starting insights generation process
[→] LOAD: Reading transformed file: Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
[✓] Loaded 687 rows for analysis
[→] ANALYZE: Generating insights and policy recommendations
[✓] Insights report generated
[→] AI_ENHANCE: Generating dual-API AI summary (Groq + HuggingFace)
[✓] AI summary added with narrative and mathematical analysis
[→] SAVE: Saving insights report
[✓] Successfully generated insights report
   📄 Insights report saved: outputs\insights_report.md

Quick Insights Summary:
   • Total illiterates analyzed: 1,049
   • High-risk wards identified: 15
   • Gender disparity detected: 62.3% female
   • Policy recommendations: 8 actionable items
```

### 5. ML Clustering Command
Analyze ward clustering patterns using SimpleMlAgent.

```bash
python cli.py cluster data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
```

**Expected Output:**
```
[→] CLUSTER: Starting ML clustering analysis
[→] LOAD: Reading transformed data for clustering
[✓] Loaded 630 wards for analysis
[→] PROCESS: Applying k-means clustering (k=5)
[✓] Clustering completed - 5 clusters identified

📊 Clustering Results:
   • Cluster 1: 198 wards (avg: 869 illiterates)
   • Cluster 2: 24 wards (avg: 3,001 illiterates) - HIGH RISK
   • Cluster 3: 198 wards (avg: 465 illiterates)
   • Cluster 4: 122 wards (avg: 1,286 illiterates)
   • Cluster 5: 88 wards (avg: 1,903 illiterates)

[✓] Clustering analysis saved: outputs\clusters.json
```

### 6. Anomaly Detection Command
Detect statistical outliers in governance data.

```bash
python cli.py anomalies data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
```

**Expected Output:**
```
[→] ANOMALIES: Starting anomaly detection analysis
[→] LOAD: Reading data for outlier detection
[✓] Loaded 630 wards for analysis
[→] PROCESS: Applying IQR-based anomaly detection
[✓] Anomaly detection completed

🚨 Anomalies Detected:
   • ASWAPURAM: 2,617 illiterates (+150% deviation)
   • BAYYARAM: 3,053 illiterates (+191% deviation)
   • CHALLA SAMUDRAM: 3,323 illiterates (+217% deviation)

📊 Analysis Summary:
   • Total outliers: 3 wards
   • Threshold (Q3 + 1.5*IQR): 1,750 illiterates
   • Mean deviation: +186%

[✓] Anomaly analysis saved: outputs\anomalies.json
```

Quick Insights Summary:
   [→] Key finding: Female illiteracy is 75.7% higher than male overall.
   [→] Recommendation: Target literacy programs in wards: Nadergul, Revenue Ward No 2, Revenue Ward No 8.
```

## Command Options

### Ingest Command
```bash
python cli.py ingest <input_file> [options]
```

**Arguments:**
- `input_file`: Path to the raw CSV file

**Options:**
- `-o, --output-dir`: Directory to save standardized files (default: `data/cleaned`)

### Clean Command
```bash
python cli.py clean <input_file> [options]
```

**Arguments:**
- `input_file`: Path to the standardized CSV file

**Options:**
- `-o, --output-dir`: Directory to save cleaned files (default: `data/cleaned`)

### Transform Command
```bash
python cli.py transform <input_file> [options]
```

**Arguments:**
- `input_file`: Path to the cleaned CSV file

**Options:**
- `-o, --output-dir`: Directory to save transformed files (default: `data/cleaned`)

### Insights Command
```bash
python cli.py insights <input_file>
```

**Arguments:**
- `input_file`: Path to the transformed CSV file

## File Naming Conventions

The pipeline follows consistent naming patterns:

1. **Raw data**: `original_filename.csv`
2. **Standardized**: `original_filename_standardized.csv`
3. **Cleaned**: `original_filename_standardized_cleaned.csv`
4. **Transformed**: `original_filename_cleaned_transformed.csv`

## Troubleshooting

### Common Issues

#### 1. Virtual Environment Not Activated
**Error:** `'python' is not recognized as an internal or external command`

**Solution:**
```bash
.\venv\Scripts\activate
```

#### 2. Missing Dependencies
**Error:** `ModuleNotFoundError: No module named 'pandas'`

**Solution:**
```bash
pip install -r requirements.txt
```

#### 3. File Not Found
**Error:** `Input file 'filename.csv' not found`

**Solution:**
- Check file path is correct
- Ensure file exists in the specified directory
- Use absolute paths if relative paths fail

#### 4. Permission Errors
**Error:** `PermissionError: [Errno 13] Permission denied`

**Solution:**
- Close any Excel files that might have the CSV open
- Run command prompt as administrator
- Check file permissions

#### 5. Encoding Issues
**Error:** `UnicodeDecodeError: 'charmap' codec can't decode`

**Solution:**
- Ensure CSV files are saved in UTF-8 encoding
- Check for special characters in data

### Performance Tips

#### For Large Datasets (>100K rows)
- Increase available RAM
- Process in chunks using custom batch sizes
- Monitor disk space in outputs directory

#### For Slow Processing
- Close unnecessary applications
- Use SSD storage for faster I/O
- Check antivirus software isn't scanning files

### Getting Help

#### CLI Help
```bash
python cli.py --help
python cli.py ingest --help
python cli.py clean --help
python cli.py transform --help
python cli.py insights --help
```

#### Debug Mode
Add verbose logging by modifying the CLI commands to include debug prints.

#### Log Files
Check the console output for detailed error messages and progress indicators.
