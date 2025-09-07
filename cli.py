#!/usr/bin/env python3
"""
RTGS CLI - A command-line interface for data processing pipeline.
"""
import json
import argparse
from pathlib import Path
import pandas as pd
import sys
import os
from dotenv import load_dotenv

# Disable Plotly by default to avoid import hanging issues
# Set ENABLE_PLOTLY=true in environment to enable interactive charts
try:
    if os.getenv('ENABLE_PLOTLY', 'false').lower() == 'true':
        import plotly.graph_objects as go
        PLOTLY_AVAILABLE = True
        print("Info: Interactive visualizations enabled")
    else:
        PLOTLY_AVAILABLE = False
        print("Info: Interactive visualizations disabled (set ENABLE_PLOTLY=true to enable)")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    print(f"Warning: Plotly not available: {e}")
    print("Install plotly for interactive visualizations: pip install plotly kaleido")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError as e:
    GROQ_AVAILABLE = False
    print(f"Warning: Groq not available: {e}")
    print("Install groq to enable AI-powered insights: pip install groq")

try:
    import signal
    import sys
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Import timeout")
    
    # Set timeout for import (Windows compatible)
    try:
        import requests
        HUGGINGFACE_AVAILABLE = True
    except Exception as e:
        HUGGINGFACE_AVAILABLE = False
        print(f"Warning: Hugging Face API disabled - {type(e).__name__}")
        
except Exception:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: Hugging Face API disabled due to import issues")

# Load environment variables
load_dotenv()

# Import agents
from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.insights_agent import InsightsAgent
try:
    from agents.simple_ml_agent import SimpleMlAgent
    ML_AGENT_AVAILABLE = True
except ImportError:
    ML_AGENT_AVAILABLE = False
    print("Warning: SimpleMlAgent not available")

import numpy as np

def print_step(step, message):
    """Print a step with formatting."""
    print(f"[>] {step}: {message}")

def print_success(message):
    """Print a success message with formatting."""
    print(f"[+] {message}")

def print_error(message):
    """Print an error message with formatting."""
    print(f"[X] ERROR: {message}")

def generate_text_chart(df, output_path):
    """Generate a simple text-based chart as fallback."""
    try:
        # Get top 10 wards by total illiterates
        top_wards = (
            df[['wardname', 'total_illiterates_by_ward']]
            .drop_duplicates()
            .nlargest(10, 'total_illiterates_by_ward')
        )
        
        viz_text = "Top 10 Wards by Illiteracy Count\n"
        viz_text += "=" * 50 + "\n\n"
        
        for i, (_, row) in enumerate(top_wards.iterrows(), 1):
            ward_name = row['wardname'][:20]  # Truncate long names
            count = int(row['total_illiterates_by_ward'])
            bar_length = min(30, count // 100)  # Scale bar length
            bar = "█" * bar_length
            viz_text += f"{i:2}. {ward_name:<20} {count:>6,} {bar}\n"
        
        # Save as text file
        text_path = str(output_path).replace('.png', '_chart.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(viz_text)
        
        print_success(f"Text-based chart saved: {text_path}")
        
    except Exception as e:
        print(f"   Text chart generation failed: {str(e)[:100]}...")

def generate_visualization(df, output_path):
    """Generate visualization with robust fallback to text-based charts."""
    try:
        # Get top 5 wards by total illiterates
        top_wards = (
            df[['wardname', 'total_illiterates_by_ward']]
            .drop_duplicates()
            .nlargest(5, 'total_illiterates_by_ward')
        )
        
        if PLOTLY_AVAILABLE:
            try:
                # Convert to simple lists to avoid pandas/plotly compatibility issues
                ward_names = top_wards['wardname'].tolist()
                illiteracy_counts = top_wards['total_illiterates_by_ward'].tolist()
                
                # Create interactive bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=ward_names,
                        y=illiteracy_counts,
                        marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                        text=[f'{int(val):,}' for val in illiteracy_counts],
                        textposition='auto'
                    )
                ])
                
                fig.update_layout(
                    title='Top 5 Wards by Illiteracy Count',
                    xaxis_title='Ward Name',
                    yaxis_title='Total Illiterates',
                    xaxis_tickangle=-45
                )
                
                # Save as HTML
                html_path = output_path.with_suffix('.html')
                fig.write_html(str(html_path))
                print_success(f"Interactive visualization saved: {html_path}")
                
                # Try to save as PNG if kaleido is available
                try:
                    fig.write_image(str(output_path))
                    print_success(f"Static visualization saved: {output_path}")
                except Exception as e:
                    print(f"   PNG export failed (install kaleido): {e}")
                    
            except (ImportError, KeyboardInterrupt, Exception) as e:
                print(f"   Plotly error ({type(e).__name__}), using text fallback")
                generate_text_chart(df, output_path)
        else:
            # Fallback to text-based visualization
            generate_text_chart(df, output_path)
            
    except Exception as e:
        print(f"   Visualization error: {str(e)[:100]}...")
        generate_text_chart(df, output_path)

def generate_ai_summary_with_groq(df, summary_stats):
    """Generate AI-powered summary using Groq LLM."""
    if not GROQ_AVAILABLE:
        return "## 📖 Narrative Insights (Local)\n\nGroq API not available. Using local analysis."
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        return "## 📖 Narrative Insights (Local)\n\nGroq API key not found in environment variables."
    
    try:
        # Add timeout and better error handling
        import socket
        socket.setdefaulttimeout(10)  # 10 second timeout
        
        client = Groq(api_key=api_key, timeout=10.0)
        
        # Check if this is governance data before accessing ward columns
        if 'wardname' not in df.columns:
            # For non-governance data, create simple summary
            numeric_cols = df.select_dtypes(include=['number']).columns
            data_summary = f"Dataset with {len(df)} records and {len(numeric_cols)} numeric columns"
            if len(numeric_cols) > 0:
                total_values = df[numeric_cols].sum().sum()
                data_summary += f". Total numeric values: {total_values}"
            
            prompt = f"""Based on this general dataset analysis:
{data_summary}

Column information: {list(df.columns)}
Summary Statistics: {summary_stats}

Write a brief narrative insight about this data, focusing on:
1. Overall data characteristics
2. Key patterns or trends visible
3. Potential applications or insights

Keep it concise and professional."""
        else:
            # Prepare context from governance data
            top_wards = df.groupby('wardname')['total_illiterates_by_ward'].first().nlargest(5)
            gender_stats = df.groupby('wardname').agg({'male': 'sum', 'female': 'sum'}).reset_index()
            gender_stats['female_ratio'] = gender_stats['female'] / (gender_stats['male'] + gender_stats['female'])
            high_female_wards = gender_stats[gender_stats['female_ratio'] > 0.6]['wardname'].tolist()
            
            prompt = f"""Based on this illiteracy data analysis:

Summary Statistics: {summary_stats}

Top 5 wards with highest illiteracy:
{chr(10).join([f"- {ward}: {count:.0f} illiterates" for ward, count in top_wards.items()])}

Wards with high female illiteracy (>60%): {', '.join(high_female_wards[:3]) if high_female_wards else 'None identified'}

Write narrative insights that include:
1. A compelling opening statement about the overall situation"""
        
            prompt += """
2. Specific examples with ward names and numbers (like "Ward X has the highest illiteracy at Y%")
3. Gender-specific observations and trends
4. Geographic or demographic patterns you notice
5. Human impact and policy implications

Use storytelling techniques. Make it engaging and accessible to policymakers. Include specific numbers and ward names for credibility. Keep under 400 words."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert policy analyst who excels at turning data into compelling human stories that drive policy action."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )
        
        return f"## 📖 Narrative Insights (Groq LLM)\n\n{response.choices[0].message.content}"
        
    except (Exception, KeyboardInterrupt) as e:
        # Fallback to local analysis when Groq fails
        print(f"[!] Groq API failed ({str(e)[:50]}...), using local analysis")
        return generate_local_fallback_insights(df)

def generate_local_fallback_insights(df):
    """Generate local narrative insights when Groq API fails."""
    if 'wardname' in df.columns:
        # Governance data fallback
        total_illiterates = int(df['total_illiterates'].sum())
        top_ward = df.groupby('wardname')['total_illiterates_by_ward'].first().idxmax()
        top_count = int(df.groupby('wardname')['total_illiterates_by_ward'].first().max())
        
        return f"""## 📖 Narrative Insights (Local Analysis)

**Key Findings from Governance Data:**

The analysis reveals {total_illiterates:,} total illiterates across {df['wardname'].nunique()} wards. The ward with highest illiteracy is **{top_ward}** with {top_count:,} illiterates.

**Gender Analysis:** Female illiteracy ({int(df['female'].sum()):,}) {'exceeds' if df['female'].sum() > df['male'].sum() else 'is lower than'} male illiteracy ({int(df['male'].sum()):,}).

**Policy Recommendations:**
- Focus intervention programs on high-burden wards
- Implement gender-specific literacy initiatives
- Establish community-based education centers in affected areas"""
    else:
        # General data fallback
        numeric_cols = df.select_dtypes(include=['number']).columns
        return f"""## 📖 Narrative Insights (Local Analysis)

**Dataset Overview:**

This dataset contains {len(df)} records with {len(df.columns)} columns, including {len(numeric_cols)} numeric fields.

**Key Characteristics:**
- Total records analyzed: {len(df)}
- Data completeness: {(df.count().sum() / (len(df) * len(df.columns)) * 100):.1f}%
- Numeric data available for trend analysis

**Applications:**
- Suitable for statistical analysis and reporting
- Can support decision-making processes
- Enables performance tracking over time"""

def generate_huggingface_analysis(df, summary_stats):
    """Generate mathematical analysis using Hugging Face API with improved error handling."""
    if not HUGGINGFACE_AVAILABLE:
        return generate_local_mathematical_analysis(df)
    
    hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_api_key:
        # Fallback to local mathematical analysis without HF API
        return generate_local_mathematical_analysis(df)
    
    try:
        # Check if this is governance data before accessing ward columns
        if 'wardname' not in df.columns:
            return generate_local_mathematical_analysis(df)
            
        # Prepare statistical data for analysis
        ward_stats = df.groupby('wardname').agg({
            'male': 'sum',
            'female': 'sum', 
            'total_illiterates_by_ward': 'first'
        }).reset_index()
        
        # Calculate key metrics
        total_wards = len(ward_stats)
        avg_illiteracy = ward_stats['total_illiterates_by_ward'].mean()
        std_illiteracy = ward_stats['total_illiterates_by_ward'].std()
        
        # Gender analysis
        ward_stats['gender_ratio'] = ward_stats['female'] / (ward_stats['male'] + ward_stats['female'])
        high_female_wards = ward_stats[ward_stats['gender_ratio'] > 0.6]
        
        # Simple clustering without sklearn (using our SimpleMlAgent approach)
        try:
            from agents.simple_ml_agent import SimpleMlAgent
            ml_agent = SimpleMlAgent()
            
            # Prepare data for clustering
            data_dict = {"combined_data": ward_stats}
            cluster_results = ml_agent.perform_clustering_analysis(data_dict)
            
            # Outlier detection using IQR method
            Q1 = ward_stats['total_illiterates_by_ward'].quantile(0.25)
            Q3 = ward_stats['total_illiterates_by_ward'].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            outliers = ward_stats[ward_stats['total_illiterates_by_ward'] > outlier_threshold]
            
            # Generate mathematical analysis report
            analysis = f"""## 🔢 Mathematical Analysis (Enhanced)

### Statistical Overview
- **Total Wards Analyzed**: {total_wards:,}
- **Mean Illiteracy**: {avg_illiteracy:.1f} ± {std_illiteracy:.1f}
- **Coefficient of Variation**: {(std_illiteracy/avg_illiteracy)*100:.1f}%

### Cluster Analysis (ML-Powered)
"""
            
            if cluster_results.get("status") == "success":
                for cluster in cluster_results.get("cluster_summary", []):
                    analysis += f"- **Cluster {cluster['cluster_id'] + 1}**: {cluster['ward_count']} wards, avg: {cluster['avg_illiterates']:.0f}\n"
                    analysis += f"  - Top wards: {', '.join(cluster['top_wards'][:2])}\n"
            
            # Outlier analysis
            if len(outliers) > 0:
                analysis += f"\n### Outlier Detection\n"
                for _, outlier in outliers.head(3).iterrows():
                    analysis += f"- **{outlier['wardname']}**: {outlier['total_illiterates_by_ward']:.0f} illiterates "
                    analysis += f"({((outlier['total_illiterates_by_ward'] - avg_illiteracy)/avg_illiteracy)*100:+.0f}% from mean)\n"
            
            # Gender disparity analysis
            analysis += f"\n### Gender Analysis\n"
            analysis += f"- **High Female Illiteracy Wards**: {len(high_female_wards)} ({(len(high_female_wards)/total_wards)*100:.1f}%)\n"
            analysis += f"- **Average Gender Ratio**: {ward_stats['gender_ratio'].mean():.2f} (female/total)\n"
            
            return analysis
            
        except Exception as e:
            return f"## 🔢 Mathematical Analysis (Basic)\n\nML analysis error: {str(e)[:100]}..."
    
    except Exception as e:
        return f"## 🔢 Mathematical Analysis (Error)\n\nAnalysis failed: {str(e)[:100]}..."

def generate_local_mathematical_analysis(df: pd.DataFrame) -> str:
    """Generate local mathematical analysis without external APIs."""
    try:
        ward_stats = df.groupby('wardname').agg({
            'male': 'sum',
            'female': 'sum', 
            'total_illiterates_by_ward': 'first'
        }).reset_index()
        
        total_wards = len(ward_stats)
        avg_illiteracy = ward_stats['total_illiterates_by_ward'].mean()
        std_illiteracy = ward_stats['total_illiterates_by_ward'].std()
        
        # Gender analysis
        ward_stats['gender_ratio'] = ward_stats['female'] / (ward_stats['male'] + ward_stats['female'])
        high_female_wards = ward_stats[ward_stats['gender_ratio'] > 0.6]
        
        analysis = f"""## 🔢 Mathematical Analysis (Local)

### Statistical Overview
- **Total Wards Analyzed**: {total_wards:,}
- **Mean Illiteracy**: {avg_illiteracy:.1f} ± {std_illiteracy:.1f}
- **Coefficient of Variation**: {(std_illiteracy/avg_illiteracy)*100:.1f}%

### Gender Analysis
- **High Female Illiteracy Wards**: {len(high_female_wards)} ({(len(high_female_wards)/total_wards)*100:.1f}%)
- **Average Gender Ratio**: {ward_stats['gender_ratio'].mean():.2f} (female/total)

### Top 5 Wards by Illiteracy
"""
        
        top_wards = ward_stats.nlargest(5, 'total_illiterates_by_ward')
        for _, ward in top_wards.iterrows():
            analysis += f"- **{ward['wardname']}**: {ward['total_illiterates_by_ward']:.0f} illiterates\n"
        
        return analysis
        
    except Exception as e:
        return f"## 🔢 Mathematical Analysis (Error)\n\nLocal analysis failed: {str(e)[:100]}..."

def generate_combined_ai_analysis(df, summary_stats):
    """Generate combined AI analysis using both Groq and HuggingFace."""
    print_step("AI ANALYSIS", "Generating narrative insights with Groq LLM")
    groq_analysis = generate_ai_summary_with_groq(df, summary_stats)
    
    print_step("AI ANALYSIS", "Generating mathematical insights with HuggingFace")
    hf_analysis = generate_huggingface_analysis(df, summary_stats)
    
    # Combine both analyses
    combined_analysis = f"""# 🤖 AI-Powered Data Insights Report

{groq_analysis}

---

{hf_analysis}

---

## 📋 Summary
This analysis combines narrative storytelling (Groq LLM) with mathematical analysis (HuggingFace/Local ML) to provide comprehensive insights for policy decision-making.
"""
    
    return combined_analysis

# CLI command functions
def ingest(args):
    """Ingest and standardize a CSV file."""
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print_error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    print_step("INGEST", f"Starting ingestion of {input_path.name}")
    
    agent = IngestionAgent()
    result = agent.ingest_csv(str(input_path))
    
    if result['status'] == 'success':
        print_success(f"Ingestion completed: {result['output_file']}")
        print(f"   Schema: {result['schema_summary']}")
    else:
        print_error(f"Ingestion failed: {result['message']}")
        sys.exit(1)

def clean(args):
    """Clean a CSV file."""
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print_error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    print_step("CLEAN", f"Starting cleaning of {input_path.name}")
    
    agent = CleaningAgent()
    result = agent.clean_csv(str(input_path))
    
    if result['status'] == 'success':
        print_success(f"Cleaning completed: {result['output_file']}")
        print(f"   Report: {result['report_file']}")
    else:
        print_error(f"Cleaning failed: {result['message']}")
        sys.exit(1)

def transform(args):
    """Transform a cleaned CSV file."""
    print_step("TRANSFORM", "Starting data transformation process")
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print_error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)

    print_step("LOAD", f"Reading cleaned file: {input_path.name}")
    df = pd.read_csv(input_path)
    print_success(f"Loaded {len(df)} rows for transformation")

    print_step("TRANSFORM", "Applying transformations")
    agent = TransformationAgent()
    transformed_df, _ = agent.transform_data(df, input_path.name)
    print_success("Data transformation completed")

    # Generate visualization
    print_step("VISUALIZE", "Generating data visualization")
    viz_path = Path("outputs/illiteracy_visualization.png")
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    generate_visualization(transformed_df, viz_path)

    # Save transformed data
    output_path = Path("data/cleaned") / f"{input_path.stem}_transformed.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transformed_df.to_csv(output_path, index=False)
    print_success(f"Transformed data saved: {output_path}")

def insights(args):
    """Generate insights from a transformed CSV file."""
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print_error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    print_step("INSIGHTS", f"Generating insights from {input_path.name}")
    
    # Load data
    df = pd.read_csv(input_path)
    print_success(f"Loaded {len(df)} rows for analysis")
    
    # Generate summary statistics
    summary_stats = {
        'total_records': len(df),
        'total_wards': df['wardname'].nunique() if 'wardname' in df.columns else 0,
        'avg_illiteracy': df['total_illiterates_by_ward'].mean() if 'total_illiterates_by_ward' in df.columns else 0
    }
    
    # Generate AI-powered insights
    print_step("AI INSIGHTS", "Generating comprehensive AI analysis")
    ai_insights = generate_combined_ai_analysis(df, summary_stats)
    
    # Save insights report
    output_path = Path("outputs/insights_report.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(ai_insights)
    
    print_success(f"Insights report saved: {output_path}")

def cluster(args):
    """Cluster wards across multiple CSV files using ML."""
    if not ML_AGENT_AVAILABLE:
        print_error("SimpleMlAgent not available. Cannot perform clustering.")
        sys.exit(1)
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print_error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    print_step("CLUSTER", f"Starting ML clustering analysis on {input_path.name}")
    
    # Load data
    df = pd.read_csv(input_path)
    print_success(f"Loaded {len(df)} rows for clustering")
    
    # Perform clustering
    ml_agent = SimpleMlAgent()
    data_list = df.to_dict('records')
    
    result = ml_agent.perform_clustering_analysis(data_list)
    
    if result and result.get('status') == 'success':
        # Save clustering results
        output_path = Path("outputs/clusters.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print_success(f"Clustering completed: {output_path}")
    else:
        error_msg = result.get('message', 'Unknown error')
        if 'traceback' in result:
            print_error(f"Clustering failed: {error_msg}")
            print("Full traceback:")
            print(result['traceback'])
        else:
            print_error(f"Clustering failed: {error_msg}")
        sys.exit(1)

def anomalies(args):
    """Detect anomalous wards using ML."""
    if not ML_AGENT_AVAILABLE:
        print_error("SimpleMlAgent not available. Cannot perform anomaly detection.")
        sys.exit(1)
    
    input_path = Path(args.input_file)
    
    if not input_path.exists():
        print_error(f"Input file '{args.input_file}' not found.")
        sys.exit(1)
    
    print_step("ANOMALIES", f"Starting anomaly detection on {input_path.name}")
    
    # Load data
    df = pd.read_csv(input_path)
    print_success(f"Loaded {len(df)} rows for anomaly detection")
    
    # Perform anomaly detection
    ml_agent = SimpleMlAgent()
    data_dict = df.to_dict('records')
    
    result = ml_agent.detect_anomalies(data_dict)
    
    if result and result.get('status') == 'success':
        # Save anomaly results
        output_path = Path("outputs/anomalies.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print_success(f"Anomaly detection completed: {output_path}")
        print(f"   Found {len(result.get('anomalies', []))} anomalies")
    else:
        print_error(f"Anomaly detection failed: {result.get('message', 'Unknown error')}")

def run_pipeline(args):
    """Run complete RTGS pipeline from raw CSV to final insights and ML analytics."""
    input_file = args.input_file
    skip_ml = getattr(args, 'skip_ml', False)
    
    print("=" * 50)
    print("RTGS CLI - Complete Pipeline Execution")
    print("=" * 50)
    print("Production-Ready Governance Analytics")
    print("Dual-API AI Integration + ML Analytics")
    print("=" * 50)
    print()
    
    # Step 1: Ingest
    print_step("PIPELINE", "Step 1: Data Ingestion")
    try:
        ingest_args = argparse.Namespace(input_file=input_file)
        ingest(ingest_args)
    except SystemExit:
        print_error("Pipeline failed at ingestion step")
        return
    
    # Determine file paths for subsequent steps
    input_path = Path(input_file)
    standardized_file = f"data/raw/{input_path.stem}_standardized.csv"
    cleaned_file = f"data/cleaned/{input_path.stem}_standardized_cleaned.csv"
    transformed_file = f"data/cleaned/{input_path.stem}_standardized_cleaned_transformed.csv"
    
    # Step 2: Clean
    print_step("PIPELINE", "Step 2: Data Cleaning")
    try:
        clean_args = argparse.Namespace(input_file=standardized_file)
        clean(clean_args)
    except SystemExit:
        print_error("Pipeline failed at cleaning step")
        return
    
    # Step 3: Transform
    print_step("PIPELINE", "Step 3: Data Transformation")
    try:
        transform_args = argparse.Namespace(input_file=cleaned_file)
        transform(transform_args)
    except SystemExit:
        print_error("Pipeline failed at transformation step")
        return
    
    # Step 4: Insights
    print_step("PIPELINE", "Step 4: AI Insights Generation")
    try:
        insights_args = argparse.Namespace(input_file=transformed_file)
        insights(insights_args)
    except SystemExit:
        print_error("Pipeline failed at insights step")
        return
    
    # Step 5 & 6: ML Analytics (if not skipped)
    if not skip_ml and ML_AGENT_AVAILABLE:
        print_step("PIPELINE", "Step 5: ML Clustering Analysis")
        try:
            cluster_args = argparse.Namespace(input_file=transformed_file)
            cluster(cluster_args)
        except SystemExit:
            print_error("Pipeline failed at clustering step")
            return
        
        print_step("PIPELINE", "Step 6: Anomaly Detection")
        try:
            anomaly_args = argparse.Namespace(input_file=transformed_file)
            anomalies(anomaly_args)
        except SystemExit:
            print_error("Pipeline failed at anomaly detection step")
            return
    elif skip_ml:
        print("   ML analytics skipped (--skip-ml flag)")
    else:
        print("   ML analytics skipped (SimpleMlAgent not available)")
    
    # Success summary
    print()
    print("=" * 50)
    print("[✓] Complete Pipeline Execution Finished!")
    print("=" * 50)
    print("Analytics Results Generated:")
    print("   Schema Analysis: outputs/schema_summary.json")
    print("   Data Quality: outputs/cleaning_report.json")
    print("   Transformations: outputs/transformation_report.json")
    print("   AI Insights: outputs/insights_report.md")
    print("   ML Clusters: outputs/clusters.json")
    print("   Anomalies: outputs/anomalies.json")
    print("   Visualizations: outputs/trend.png")
    print()
    print("Production-ready governance analytics complete!")
    print("Complete data pipeline executed successfully")
    print("Dual-API AI analysis (Groq + HuggingFace)")
    if not skip_ml and ML_AGENT_AVAILABLE:
        print("ML analytics completed (clustering + anomaly detection)")
    print()

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="RTGS CLI - Data Processing Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest and standardize a CSV file.')
    ingest_parser.add_argument('input_file', help='Path to the input CSV file.')
    ingest_parser.set_defaults(func=ingest)
    
    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean a CSV file.')
    clean_parser.add_argument('input_file', help='Path to the input CSV file.')
    clean_parser.set_defaults(func=clean)
    
    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform a cleaned CSV file.')
    transform_parser.add_argument('input_file', help='Path to the cleaned CSV file.')
    transform_parser.set_defaults(func=transform)
    
    # Insights command
    insights_parser = subparsers.add_parser('insights', help='Generate insights from a transformed CSV file.')
    insights_parser.add_argument('input_file', help='Path to the transformed CSV file.')
    insights_parser.set_defaults(func=insights)
    
    # Cluster command
    cluster_parser = subparsers.add_parser('cluster', help='Cluster wards across multiple CSV files using ML.')
    cluster_parser.add_argument('input_file', help='Path to the CSV file for clustering.')
    cluster_parser.set_defaults(func=cluster)
    
    # Anomalies command
    anomalies_parser = subparsers.add_parser('anomalies', help='Detect anomalous wards using ML.')
    anomalies_parser.add_argument('input_file', help='Path to the CSV file for anomaly detection.')
    anomalies_parser.set_defaults(func=anomalies)
    
    # Pipeline command - run entire pipeline with single command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete RTGS pipeline from raw CSV to final insights and ML analytics.')
    pipeline_parser.add_argument('input_file', help='Path to the raw CSV file.')
    pipeline_parser.add_argument('--skip-ml', action='store_true', help='Skip ML clustering and anomaly detection.')
    pipeline_parser.set_defaults(func=run_pipeline)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)

if __name__ == "__main__":
    main()
