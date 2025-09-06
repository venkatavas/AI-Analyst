#!/usr/bin/env python3
"""
Batch processor for all CSV files in the RTGS CLI project.
Processes all CSV files through the complete pipeline automatically.
"""

import os
import subprocess
import glob
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"[→] {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"[✓] {description} completed")
            return True
        else:
            print(f"[✗] {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[✗] {description} error: {e}")
        return False

def process_csv_file(csv_file):
    """Process a single CSV file through the complete pipeline."""
    file_name = Path(csv_file).stem
    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"{'='*60}")
    
    # Step 1: Ingest
    if not run_command(f'python cli.py ingest "{csv_file}"', f"Ingesting {file_name}"):
        return False
    
    # Step 2: Clean
    standardized_file = f"data/cleaned/{file_name}_standardized.csv"
    if not run_command(f'python cli.py clean "{standardized_file}"', f"Cleaning {file_name}"):
        return False
    
    # Step 3: Transform
    cleaned_file = f"data/cleaned/{file_name}_standardized_cleaned.csv"
    if not run_command(f'python cli.py transform "{cleaned_file}"', f"Transforming {file_name}"):
        return False
    
    # Step 4: Insights
    transformed_file = f"data/cleaned/{file_name}_cleaned_transformed.csv"
    if not run_command(f'python cli.py insights "{transformed_file}"', f"Generating insights for {file_name}"):
        return False
    
    print(f"[✓] {file_name} processing completed successfully!")
    return True

def main():
    """Process all CSV files in data/raw directory."""
    print("🚀 RTGS CLI - Batch CSV Processor")
    print("Processing all CSV files through complete pipeline\n")
    
    # Find all CSV files in data/raw
    csv_files = glob.glob("data/raw/*.csv")
    
    if not csv_files:
        print("[✗] No CSV files found in data/raw directory")
        return
    
    print(f"Found {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"  • {Path(csv_file).name}")
    
    print("\nStarting batch processing...\n")
    
    successful = 0
    failed = 0
    
    for csv_file in csv_files:
        if process_csv_file(csv_file):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {len(csv_files)}")
    
    if successful > 0:
        print(f"\n🎉 Successfully processed {successful} datasets!")
        print("📁 Check outputs/ directory for generated reports")
        print("📁 Check data/cleaned/ directory for processed data")
        
        # Run clustering and anomaly detection on all processed files
        print(f"\n[→] Running ML analysis on all processed datasets...")
        cleaned_files = glob.glob("data/cleaned/*_cleaned_transformed.csv")
        if cleaned_files:
            files_str = " ".join([f'"{f}"' for f in cleaned_files])
            run_command(f'python cli.py cluster {files_str}', "Clustering analysis")
            run_command(f'python cli.py anomalies {files_str}', "Anomaly detection")

if __name__ == "__main__":
    main()
