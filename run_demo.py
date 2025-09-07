#!/usr/bin/env python3
"""
RTGS CLI - Complete Pipeline Demo Script
Production-Ready Governance Analytics with Dual-API AI Integration + ML Analytics
"""

import subprocess
import sys
import os
from pathlib import Path
import glob

def run_command(description, command):
    """Execute a CLI command with error handling"""
    print(f"[>] {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"[+] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] Error in {description}")
        print(f"Error output: {e.stderr}")
        return False

def backup_outputs(dataset_name):
    """Backup current outputs to dataset-specific folder."""
    outputs_dir = Path("outputs")
    backup_dir = Path("outputs") / f"{dataset_name}_results"
    backup_dir.mkdir(exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "schema_summary.json", "schema_summary.csv", "cleaning_report.json",
        "transformation_report.json", "insights_report.md", "clusters.json", 
        "anomalies.json", "trend.png", "trend_chart.txt", "illiteracy_visualization_chart.txt"
    ]
    
    for filename in files_to_backup:
        source = outputs_dir / filename
        if source.exists():
            import shutil
            shutil.copy2(source, backup_dir / filename)

def process_dataset(csv_file):
    """Process a single dataset through the complete pipeline."""
    csv_path = Path(csv_file)
    base_name = csv_path.stem.replace('.csv', '')
    print(f"\n{'='*60}")
    print(f"Processing Dataset: {base_name}")
    print(f"{'='*60}")
    
    # Pipeline steps for this dataset
    steps = [
        (f"Step 1: Ingest {base_name} dataset", 
         f'python cli.py ingest "data/raw/{csv_path.name}"'),
        
        (f"Step 2: Clean {base_name} standardized dataset", 
         f'python cli.py clean "data/raw/{base_name}_standardized.csv"'),
        
        (f"Step 3: Transform {base_name} cleaned dataset", 
         f'python cli.py transform "data/cleaned/{base_name}_standardized_cleaned.csv"'),
        
        (f"Step 4: Generate AI insights for {base_name}", 
         f'python cli.py insights "data/cleaned/{base_name}_standardized_cleaned_transformed.csv"'),
        
        (f"Step 5: ML clustering analysis for {base_name}", 
         f'python cli.py cluster "data/cleaned/{base_name}_standardized_cleaned_transformed.csv"'),
        
        (f"Step 6: Anomaly detection for {base_name}", 
         f'python cli.py anomalies "data/cleaned/{base_name}_standardized_cleaned_transformed.csv"')
    ]
    
    # Execute pipeline steps for this dataset
    for description, command in steps:
        if not run_command(description, command):
            print(f"[X] Pipeline failed for {base_name} at: {description}")
            return False
        print()
    
    # Backup outputs for this dataset
    backup_outputs(base_name)
    print(f"[+] {base_name} processing completed successfully!")
    return True

def main():
    """Run complete RTGS CLI pipeline on all CSV datasets"""
    
    print("=" * 60)
    print("RTGS CLI - Multi-Dataset Pipeline Demo")
    print("=" * 60)
    print("Production-Ready Governance Analytics")
    print("AI Integration + ML Analytics for All Datasets")
    print("=" * 60)
    print()
    
    # Find all CSV files in data/raw directory (excluding processed files)
    csv_files = glob.glob("data/raw/*.csv")
    
    # Filter out already processed files (those with _standardized suffix)
    # Also exclude incompatible datasets that don't match governance schema
    original_files = [f for f in csv_files if not f.endswith('_standardized.csv')]
    
    # Include all datasets - let the pipeline handle different schemas gracefully
    # No filtering needed - process all available datasets
    
    if not original_files:
        print("[X] No CSV files found in data/raw directory")
        print("Please ensure CSV datasets exist before running the demo.")
        return False
    
    print(f"Found {len(original_files)} datasets to process:")
    for i, file in enumerate(original_files, 1):
        print(f"  {i}. {Path(file).name}")
    print()
    
    # Process each dataset through the complete pipeline
    successful_datasets = []
    failed_datasets = []
    
    for data_file in original_files:
        if process_dataset(data_file):
            successful_datasets.append(Path(data_file).name)
        else:
            failed_datasets.append(Path(data_file).name)
    
    # Final summary
    print("\n" + "=" * 60)
    print("[+] Multi-Dataset Pipeline Demo Finished!")
    print("=" * 60)
    
    if successful_datasets:
        print(f"Successfully Processed ({len(successful_datasets)} datasets):")
        for dataset in successful_datasets:
            print(f"   [+] {dataset}")
    
    if failed_datasets:
        print(f"\nFailed Processing ({len(failed_datasets)} datasets):")
        for dataset in failed_datasets:
            print(f"   [X] {dataset}")
    
    print("\nAnalytics Results Generated (per dataset):")
    print("   Schema Analysis: outputs/schema_summary.json")
    print("   Data Quality: outputs/cleaning_report.json")
    print("   Transformations: outputs/transformation_report.json")
    print("   AI Insights: outputs/insights_report.md")
    print("   ML Clusters: outputs/clusters.json")
    print("   Anomalies: outputs/anomalies.json")
    print("   Visualizations: outputs/trend.png")
    print()
    print("Production-ready governance analytics complete!")
    print(f"Multi-dataset pipeline executed on {len(original_files)} datasets")
    print("AI analysis (Groq LLM + Local ML)")
    print("ML analytics completed (clustering + anomaly detection)")
    print()
    print("Ready for comprehensive policy decision-making!")
    
    # Return success only if all datasets processed successfully
    return len(failed_datasets) == 0

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nDemo completed successfully!")
        else:
            print("\nDemo failed. Check error messages above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[X] Unexpected error: {e}")
        sys.exit(1)
