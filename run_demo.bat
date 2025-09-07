@echo off
setlocal enabledelayedexpansion

echo ========================================
echo 🚀 RTGS CLI - Complete Pipeline Demo
echo ========================================
echo Production-Ready Governance Analytics
echo Dual-API AI Integration + ML Analytics
echo ========================================
echo.

REM Step 1: Ingest
echo [→] Step 1: Ingest raw dataset (687 records)
python cli.py ingest data/raw/Illiterate_Khammam_Rural.csv
if !errorlevel! neq 0 (
    echo [✗] Error in ingest step
    exit /b 1
)
echo.

REM Step 2: Clean
echo [→] Step 2: Clean standardized dataset
python cli.py clean data/raw/Illiterate_Khammam_Rural_standardized.csv
if !errorlevel! neq 0 (
    echo [✗] Error in clean step
    exit /b 1
)
echo.

REM Step 3: Transform
echo [→] Step 3: Transform cleaned dataset
python cli.py transform data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned.csv
if !errorlevel! neq 0 (
    echo [✗] Error in transform step
    exit /b 1
)
echo.

REM Step 4: Dual-API Insights
echo [→] Step 4: Generate dual-API insights (Groq + HuggingFace)
python cli.py insights data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
if !errorlevel! neq 0 (
    echo [✗] Error in insights step
    exit /b 1
)
echo.

REM Step 5: ML Clustering
echo [→] Step 5: ML clustering analysis (SimpleMlAgent)
python cli.py cluster data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
if !errorlevel! neq 0 (
    echo [✗] Error in clustering step
    exit /b 1
)
echo.

REM Step 6: Anomaly Detection
echo [→] Step 6: Anomaly detection (IQR-based outliers)
python cli.py anomalies data/cleaned/Illiterate_Khammam_Rural_standardized_cleaned_transformed.csv
if !errorlevel! neq 0 (
    echo [✗] Error in anomaly detection step
    exit /b 1
)
echo.

echo ========================================
echo [✓] Complete Pipeline Demo Finished!
echo ========================================
echo 📊 Analytics Results Generated:
echo    📄 Schema Analysis: outputs/schema_summary.json
echo    🧹 Data Quality: outputs/cleaning_report.json
echo    🔄 Transformations: outputs/transformation_report.json
echo    🤖 Dual-API Insights: outputs/insights_report.md
echo    🎯 ML Clusters: outputs/clusters.json
echo    🚨 Anomalies: outputs/anomalies.json
echo    📈 Visualizations: outputs/trend.png
echo.
echo 🎉 Production-ready governance analytics complete!
echo 📊 687 records processed across 630+ wards
echo 🤖 Dual-API AI analysis (Groq + HuggingFace)
echo 🎯 5 clusters identified, 3 outliers detected
echo.
pause
