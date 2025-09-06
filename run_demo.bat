@echo off
setlocal enabledelayedexpansion

echo [→] Starting Real-Time Governance System demo...
echo.

REM Step 1: Ingest
echo [→] Step 1: Ingest raw dataset
.\venv\Scripts\python.exe cli.py ingest data/raw/Illiterate_Rangareddy_Urban_Area.csv
if !errorlevel! neq 0 (
    echo [✗] Error in ingest step
    exit /b 1
)
echo.

REM Step 2: Clean
echo [→] Step 2: Clean standardized dataset
.\venv\Scripts\python.exe cli.py clean data/cleaned/Illiterate_Rangareddy_Urban_Area_standardized.csv
if !errorlevel! neq 0 (
    echo [✗] Error in clean step
    exit /b 1
)
echo.

REM Step 3: Transform
echo [→] Step 3: Transform cleaned dataset
.\venv\Scripts\python.exe cli.py transform data/cleaned/Illiterate_Rangareddy_Urban_Area_standardized_cleaned.csv
if !errorlevel! neq 0 (
    echo [✗] Error in transform step
    exit /b 1
)
echo.

REM Step 4: Insights
echo [→] Step 4: Generate insights and recommendations
.\venv\Scripts\python.exe cli.py insights data/cleaned/Illiterate_Rangareddy_Urban_Area_cleaned_transformed.csv
if !errorlevel! neq 0 (
    echo [✗] Error in insights step
    exit /b 1
)
echo.

echo [✓] Demo complete! Check /outputs/ for reports, JSON, and visualization.
echo.
echo Generated files:
echo    📊 outputs/schema_summary.csv
echo    📄 outputs/schema_summary.json
echo    🧹 outputs/cleaning_report.json
echo    🔄 outputs/transformation_report.json
echo    📝 outputs/insights_report.md
echo    📈 outputs/trend.png
echo.
pause
