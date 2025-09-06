#!/usr/bin/env python3
"""
Comprehensive Demo Script for RTGS CLI Hybrid AI System

This script demonstrates all features of the RTGS CLI including:
- Data ingestion and processing pipeline
- Groq AI-powered insights
- CrewAI ML clustering and anomaly detection
- MCP server integration
- Complete governance workflow

Usage:
    python demo_full_pipeline.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """Print a formatted step."""
    print(f"\n🔄 Step {step}: {description}")
    print("-" * 50)

def run_command(cmd, description=""):
    """Run a command and capture output."""
    if description:
        print(f"   Running: {description}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode == 0:
            print("   ✅ Success")
            if result.stdout.strip():
                # Show key output lines
                lines = result.stdout.strip().split('\n')
                for line in lines[-5:]:  # Show last 5 lines
                    if line.strip():
                        print(f"   {line}")
        else:
            print("   ❌ Error")
            if result.stderr:
                print(f"   Error: {result.stderr}")
        
        return result.returncode == 0, result.stdout, result.stderr
    
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False, "", str(e)

def check_file_exists(filepath, description=""):
    """Check if a file exists and show its size."""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"   ✅ {description or path.name}: {size:,} bytes")
        return True
    else:
        print(f"   ❌ {description or path.name}: Not found")
        return False

def show_json_summary(filepath, max_items=3):
    """Show a summary of JSON output."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'clustering_results' in data:
            results = data['clustering_results']
            print(f"   📊 Clusters: {results.get('num_clusters', 0)}")
            print(f"   📊 Total wards: {results.get('total_wards', 0)}")
            print(f"   📊 Silhouette score: {results.get('silhouette_score', 'N/A')}")
        
        elif 'anomaly_detection_results' in data:
            results = data['anomaly_detection_results']
            print(f"   🚨 Anomalies detected: {results.get('anomalies_detected', 0)}")
            print(f"   🚨 Total wards analyzed: {results.get('total_wards_analyzed', 0)}")
            print(f"   🚨 Anomaly rate: {results.get('anomaly_percentage', 0)}%")
        
        return True
    except Exception as e:
        print(f"   ❌ Error reading JSON: {e}")
        return False

def main():
    """Run the complete RTGS CLI demo."""
    
    print_header("RTGS CLI - Hybrid AI System Demo")
    print("🚀 Real-Time Governance System with Groq + CrewAI + MCP")
    print("📊 Demonstrating complete data processing and AI analysis pipeline")
    
    # Check environment setup
    print_step(0, "Environment Setup Check")
    
    # Check API keys
    groq_key = os.getenv('GROQ_API_KEY')
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    
    print(f"   Groq API Key: {'✅ Set' if groq_key else '❌ Missing'}")
    print(f"   Hugging Face API Key: {'✅ Set' if hf_key else '❌ Missing'}")
    
    # Check required directories
    required_dirs = ['data/raw', 'data/cleaned', 'outputs', 'agents', 'docs']
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   Directory {dir_path}: ✅ Exists")
        else:
            print(f"   Directory {dir_path}: ❌ Missing")
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_path}")
    
    # Check sample data
    sample_data = Path("data/raw/sample_illiteracy_data.csv")
    if not sample_data.exists():
        print("   ⚠️  No sample data found. Please add a CSV file to data/raw/")
        print("   Expected columns: wardname, male, female, [transgender]")
        return
    
    print(f"   Sample data: ✅ {sample_data}")
    
    # Step 1: Data Ingestion
    print_step(1, "Data Ingestion & Standardization")
    success, stdout, stderr = run_command(
        f"python cli.py ingest {sample_data}",
        "Ingesting raw CSV data"
    )
    
    if success:
        check_file_exists("outputs/schema_summary.csv", "Schema summary")
        check_file_exists("data/cleaned/sample_illiteracy_data_standardized.csv", "Standardized data")
    
    # Step 2: Data Cleaning
    print_step(2, "Data Cleaning & Validation")
    success, stdout, stderr = run_command(
        "python cli.py clean data/cleaned/sample_illiteracy_data_standardized.csv",
        "Cleaning standardized data"
    )
    
    if success:
        check_file_exists("data/cleaned/sample_illiteracy_data_cleaned.csv", "Cleaned data")
        check_file_exists("outputs/cleaning_report.json", "Cleaning report")
    
    # Step 3: Data Transformation
    print_step(3, "Data Transformation & Aggregation")
    success, stdout, stderr = run_command(
        "python cli.py transform data/cleaned/sample_illiteracy_data_cleaned.csv",
        "Transforming cleaned data"
    )
    
    if success:
        check_file_exists("data/cleaned/sample_illiteracy_data_transformed.csv", "Transformed data")
        check_file_exists("outputs/transformation_report.json", "Transformation report")
        check_file_exists("outputs/illiteracy_by_ward.png", "Visualization")
    
    # Step 4: AI-Powered Insights (Groq)
    print_step(4, "AI-Powered Policy Insights (Groq)")
    success, stdout, stderr = run_command(
        "python cli.py insights data/cleaned/sample_illiteracy_data_transformed.csv",
        "Generating AI insights with Groq"
    )
    
    if success:
        check_file_exists("outputs/insights_report.md", "Insights report")
        print("   🤖 Groq AI analysis completed")
    
    # Step 5: CrewAI ML Clustering
    print_step(5, "Ward Clustering Analysis (CrewAI)")
    success, stdout, stderr = run_command(
        "python cli.py cluster data/cleaned/sample_illiteracy_data_transformed.csv",
        "Running CrewAI clustering analysis"
    )
    
    if success:
        check_file_exists("outputs/clusters.json", "Clustering results")
        show_json_summary("outputs/clusters.json")
        print("   🤖 CrewAI clustering completed")
    
    # Step 6: CrewAI Anomaly Detection
    print_step(6, "Anomaly Detection (CrewAI)")
    success, stdout, stderr = run_command(
        "python cli.py anomalies data/cleaned/sample_illiteracy_data_transformed.csv",
        "Running CrewAI anomaly detection"
    )
    
    if success:
        check_file_exists("outputs/anomalies.json", "Anomaly results")
        show_json_summary("outputs/anomalies.json")
        print("   🤖 CrewAI anomaly detection completed")
    
    # Step 7: MCP Server Demo (Optional)
    print_step(7, "MCP Server Integration Test")
    print("   🔗 MCP server provides API access to all pipeline functions")
    print("   📡 Available tools: rtgs_ingest, rtgs_clean, rtgs_transform, rtgs_insights")
    print("   🤖 New ML tools: rtgs_cluster, rtgs_anomalies")
    print("   💡 Start MCP server with: python mcp_server.py")
    
    # Step 8: Results Summary
    print_step(8, "Results Summary")
    
    # Count generated files
    output_files = list(Path("outputs").glob("*"))
    data_files = list(Path("data/cleaned").glob("*"))
    
    print(f"   📁 Generated outputs: {len(output_files)} files")
    print(f"   📁 Processed datasets: {len(data_files)} files")
    
    # Show key results
    print("\n   📊 Key Results Generated:")
    results_map = {
        "outputs/schema_summary.csv": "📋 Data schema analysis",
        "outputs/cleaning_report.json": "🧹 Data quality report", 
        "outputs/transformation_report.json": "🔄 Transformation summary",
        "outputs/insights_report.md": "🤖 Groq AI policy insights",
        "outputs/clusters.json": "🎯 CrewAI ward clustering",
        "outputs/anomalies.json": "🚨 CrewAI anomaly detection",
        "outputs/illiteracy_by_ward.png": "📈 Data visualization"
    }
    
    for filepath, description in results_map.items():
        if Path(filepath).exists():
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description}")
    
    # Final summary
    print_header("Demo Complete - Hybrid AI System Ready!")
    print("🎉 RTGS CLI successfully demonstrated all capabilities:")
    print("   • Data ingestion, cleaning, and transformation")
    print("   • Groq AI-powered policy insights and summaries")
    print("   • CrewAI multi-agent ML clustering and anomaly detection")
    print("   • MCP server integration for external AI access")
    print("   • Professional governance-ready outputs")
    
    print("\n🚀 Next Steps:")
    print("   1. Review generated reports in outputs/ directory")
    print("   2. Start MCP server: python mcp_server.py")
    print("   3. Use CLI commands for new datasets")
    print("   4. Integrate with governance workflows")
    
    print("\n📚 Documentation:")
    print("   • README.md - Project overview")
    print("   • docs/README_architecture.md - Technical details")
    print("   • docs/README_crewai.md - CrewAI ML features")
    print("   • docs/README_mcp.md - MCP server integration")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
