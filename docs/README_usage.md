# Usage Guide

## Quick Start

### One-Command Demo
```bash
.\run_demo.bat
```
This executes the complete pipeline automatically using the included sample dataset.

## Step-by-Step Usage

### 1. Ingest Command
Process raw CSV data and generate schema information.

```bash
.\venv\Scripts\python.exe cli.py ingest data/raw/Illiterate_Rangareddy_Urban_Area.csv
```

**Expected Output:**
```
[→] INGEST: Starting data ingestion process
[→] LOAD: Reading CSV file: Illiterate_Rangareddy_Urban_Area.csv
[✓] Loaded 126 rows and 7 columns
[→] STANDARDIZE: Converting column names to lowercase with underscores
[✓] Column names standardized
[→] ANALYZE: Generating schema summary

📊 Dataset Overview:
   • Total Rows: 126
   • Total Columns: 7

📋 Column Details:
   • district (object): 126 non-null (0.0% missing), 1 unique
   • mandal (object): 126 non-null (0.0% missing), 1 unique
   • muncipality (object): 126 non-null (0.0% missing), 1 unique
   • wardname (object): 126 non-null (0.0% missing), 42 unique
   • female (int64): 126 non-null (0.0% missing), 55 unique
   • male (int64): 126 non-null (0.0% missing), 48 unique
   • transgender (int64): 126 non-null (0.0% missing), 1 unique

[→] SAVE: Saving processed files
[✓] Successfully processed Illiterate_Rangareddy_Urban_Area.csv
   📁 Standardized data: data\cleaned\Illiterate_Rangareddy_Urban_Area_standardized.csv
   📄 Schema JSON: outputs\schema_summary.json
   📊 Schema CSV: outputs\schema_summary.csv
```

### 2. Clean Command
Remove duplicates, handle missing values, and detect outliers.

```bash
.\venv\Scripts\python.exe cli.py clean data/cleaned/Illiterate_Rangareddy_Urban_Area_standardized.csv
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
[✓] Successfully cleaned Illiterate_Rangareddy_Urban_Area_standardized.csv
   📁 Cleaned data: data\cleaned\Illiterate_Rangareddy_Urban_Area_standardized_cleaned.csv
   📄 Cleaning report: outputs\cleaning_report.json
```

### 3. Transform Command
Apply aggregations, create derived features, and generate visualizations.

```bash
.\venv\Scripts\python.exe cli.py transform data/cleaned/Illiterate_Rangareddy_Urban_Area_standardized_cleaned.csv
```

**Expected Output:**
```
[→] TRANSFORM: Starting data transformation process
[→] LOAD: Reading cleaned file: Illiterate_Rangareddy_Urban_Area_standardized_cleaned.csv
[✓] Loaded 126 rows for transformation
[→] PROCESS: Applying transformations and aggregations
[✓] Transformations completed

🔄 Transformation Summary:
   • Total illiterates: 6,789
   • Male illiterates: 2,567
   • Female illiterates: 4,222
   • Top ward: Nadergul (456 illiterates)

[→] SAVE: Saving transformed data and generating visualization
[✓] Successfully transformed Illiterate_Rangareddy_Urban_Area_standardized_cleaned.csv
   📁 Transformed data: data\cleaned\Illiterate_Rangareddy_Urban_Area_cleaned_transformed.csv
   📄 Transformation report: outputs\transformation_report.json
   📊 Visualization: outputs\trend.png
```

### 4. Insights Command
Generate policy recommendations and AI-enhanced summaries.

```bash
.\venv\Scripts\python.exe cli.py insights data/cleaned/Illiterate_Rangareddy_Urban_Area_cleaned_transformed.csv
```

**Expected Output:**
```
[→] INSIGHTS: Starting insights generation process
[→] LOAD: Reading transformed file: Illiterate_Rangareddy_Urban_Area_cleaned_transformed.csv
[✓] Loaded 126 rows for analysis
[→] ANALYZE: Generating insights and policy recommendations
[✓] Insights report generated
[→] AI_ENHANCE: Generating AI-powered summary (placeholder)
[✓] AI summary added
[→] SAVE: Saving insights report
[✓] Successfully generated insights report
   Insights report saved: outputs\insights_report.md

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
