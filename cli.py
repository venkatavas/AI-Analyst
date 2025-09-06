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
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, visualization will be skipped")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq not available, using placeholder AI summary")

# Load environment variables
load_dotenv()

# Import agents
from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.insights_agent import InsightsAgent
# from agents.ml_agent import MLAgent  # Temporarily disabled due to import conflicts

def print_step(step_name, message):
    """Print a formatted step message."""
    print(f"[→] {step_name}: {message}")

def print_success(message):
    """Print a success message."""
    print(f"[✓] {message}")

def print_error(message):
    """Print an error message."""
    print(f"[✗] Error: {message}", file=sys.stderr)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(r'[^\w\s]', '_', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )
    return df

def generate_schema_csv(df, schema, output_path):
    """Generate a CSV summary of the schema."""
    schema_data = []
    for col_info in schema["columns"]:
        col_name = col_info["name"]
        dtype = col_info["dtype"]
        
        # Get basic stats for numeric columns
        try:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                min_val = df[col_name].min()
                max_val = df[col_name].max()
                range_info = f"{min_val} - {max_val}"
                unit = "count" if "count" in col_name.lower() else "numeric"
            else:
                range_info = f"{col_info['unique_count']} unique values"
                unit = "categorical"
        except KeyError:
            # Handle case where column name might have changed during standardization
            range_info = f"{col_info['unique_count']} unique values"
            unit = "categorical"
        
        schema_data.append({
            "column_name": col_name,
            "data_type": dtype,
            "non_null_count": col_info["non_null_count"],
            "null_count": col_info["null_count"],
            "unique_count": col_info["unique_count"],
            "range_or_values": range_info,
            "unit_type": unit
        })
    
    schema_df = pd.DataFrame(schema_data)
    schema_df.to_csv(output_path, index=False)

def ingest(args):
    """
    Ingest and process a CSV file, then save standardized version and schema.
    """
    try:
        print_step("INGEST", "Starting data ingestion process")
        
        # Resolve input file path
        input_path = Path(args.input_file)
        if not input_path.exists():
            print_error(f"Input file '{args.input_file}' not found.")
            sys.exit(1)
        
        # Read the input file
        print_step("LOAD", f"Reading CSV file: {input_path.name}")
        df = pd.read_csv(input_path)
        print_success(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Standardize column names first
        print_step("STANDARDIZE", "Converting column names to lowercase with underscores")
        df_cleaned = standardize_columns(df)
        print_success("Column names standardized")
        
        # Generate schema summary after standardization
        print_step("ANALYZE", "Generating schema summary")
        insights = InsightsAgent()
        schema = insights.generate_schema_summary(df_cleaned)
        
        # Print schema summary
        print("\n📊 Dataset Overview:")
        print(f"   • Total Rows: {schema['total_rows']:,}")
        print(f"   • Total Columns: {schema['total_columns']}")
        print("\n📋 Column Details:")
        for col in schema["columns"]:
            null_pct = (col['null_count'] / schema['total_rows']) * 100
            print(f"   • {col['name']} ({col['dtype']}): {col['non_null_count']:,} non-null ({null_pct:.1f}% missing), {col['unique_count']:,} unique")
        
        # Prepare output file names
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{input_path.stem}_standardized.csv"
        schema_json_path = Path("outputs/schema_summary.json")
        schema_csv_path = Path("outputs/schema_summary.csv")
        
        # Save outputs
        print_step("SAVE", "Saving processed files")
        output_dir.mkdir(parents=True, exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)
        
        df_cleaned.to_csv(output_path, index=False)
        with open(schema_json_path, 'w') as f:
            json.dump(schema, f, indent=2)
        generate_schema_csv(df_cleaned, schema, schema_csv_path)
        
        print_success(f"Successfully processed {input_path.name}")
        print(f"   📁 Standardized data: {output_path}")
        print(f"   📄 Schema JSON: {schema_json_path}")
        print(f"   📊 Schema CSV: {schema_csv_path}")
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

def print_cleaning_summary(report):
    """Print a formatted cleaning summary to console."""
    print("\n🧹 Cleaning Summary:")
    print(f"   • Duplicates removed: {report['duplicates_removed']:,}")
    
    if report['missing_values_filled']:
        print(f"   • Missing values filled:")
        for col, count in report['missing_values_filled'].items():
            print(f"     - {col}: {count:,} values")
    else:
        print(f"   • Missing values filled: 0")
    
    if report['outliers_replaced']:
        print(f"   • Outliers replaced:")
        for col, count in report['outliers_replaced'].items():
            print(f"     - {col}: {count:,} values")
    else:
        print(f"   • Outliers replaced: 0")

def clean(args):
    """
    Clean a standardized CSV file.
    """
    try:
        print_step("CLEAN", "Starting data cleaning process")
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            print_error(f"Input file '{args.input_file}' not found.")
            sys.exit(1)

        print_step("LOAD", f"Reading standardized file: {input_path.name}")
        df = pd.read_csv(input_path)
        print_success(f"Loaded {len(df)} rows for cleaning")

        print_step("PROCESS", "Applying data cleaning operations")
        agent = CleaningAgent()
        df_cleaned, report = agent.clean_data(df)
        print_success(f"Cleaning completed - {len(df_cleaned)} rows remaining")

        # Print cleaning summary to console
        print_cleaning_summary(report)

        # Prepare output paths
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{input_path.stem}_cleaned.csv"
        report_path = Path("outputs/cleaning_report.json")

        # Save outputs
        print_step("SAVE", "Saving cleaned data and report")
        output_dir.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print_success(f"Successfully cleaned {input_path.name}")
        print(f"   📁 Cleaned data: {output_path}")
        print(f"   📄 Cleaning report: {report_path}")

    except Exception as e:
        print_error(str(e))
        sys.exit(1)

def generate_visualization(df, output_path):
    """Generate a bar chart of top 5 wards with highest illiteracy."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping visualization")
        return False
        
    try:
        # Get top 5 wards by total illiterates
        top_wards = (
            df[['wardname', 'total_illiterates_by_ward']]
            .drop_duplicates()
            .nlargest(5, 'total_illiterates_by_ward')
        )
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(top_wards)), top_wards['total_illiterates_by_ward'], 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        plt.title('Top 5 Wards by Illiteracy Count', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Ward Name', fontsize=12)
        plt.ylabel('Total Illiterates', fontsize=12)
        
        # Set ward names as x-axis labels
        ward_names = [name[:15] + '...' if len(name) > 15 else name for name in top_wards['wardname']]
        plt.xticks(range(len(top_wards)), ward_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"Warning: Could not generate visualization: {e}")
        return False

def transform(args):
    """
    Transform a cleaned CSV file.
    """
    try:
        print_step("TRANSFORM", "Starting data transformation process")
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            print_error(f"Input file '{args.input_file}' not found.")
            sys.exit(1)

        print_step("LOAD", f"Reading cleaned file: {input_path.name}")
        df = pd.read_csv(input_path)
        print_success(f"Loaded {len(df)} rows for transformation")

        print_step("PROCESS", "Applying transformations and aggregations")
        agent = TransformationAgent()
        df_transformed, report = agent.transform_data(df, input_path.name)
        print_success("Transformations completed")

        # Print transformation summary
        print(f"\n🔄 Transformation Summary:")
        print(f"   • Total illiterates: {report['total_illiterates']:,}")
        print(f"   • Male illiterates: {report['aggregations']['by_gender']['male']:,}")
        print(f"   • Female illiterates: {report['aggregations']['by_gender']['female']:,}")
        print(f"   • Top ward: {report['aggregations']['top_wards'][0]['ward']} ({report['aggregations']['top_wards'][0]['illiterates']:,} illiterates)")

        # Prepare output paths
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{input_path.stem.replace('_standardized', '')}_transformed.csv"
        report_path = Path("outputs/transformation_report.json")
        viz_path = Path("outputs/trend.png")

        # Save outputs
        print_step("SAVE", "Saving transformed data and generating visualization")
        output_dir.mkdir(parents=True, exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)
        
        df_transformed.to_csv(output_path, index=False)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate visualization
        viz_success = generate_visualization(df_transformed, viz_path)

        print_success(f"Successfully transformed {input_path.name}")
        print(f"   📁 Transformed data: {output_path}")
        print(f"   📄 Transformation report: {report_path}")
        if viz_success:
            print(f"   📊 Visualization: {viz_path}")

    except Exception as e:
        print_error(str(e))
        sys.exit(1)

def generate_ai_summary_with_groq(report_content, df):
    """Generate AI-powered summary using Groq API."""
    if not GROQ_AVAILABLE:
        print("Warning: Groq not available, using placeholder summary")
        return generate_ai_summary_placeholder(report_content)
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Warning: No Groq API key found, using placeholder summary")
        return generate_ai_summary_placeholder(report_content)
    
    try:
        # Initialize Groq client with timeout and error handling
        client = Groq(
            api_key=api_key,
            timeout=30.0  # 30 second timeout
        )
        
        # Prepare data context for AI analysis
        total_illiterates = int(df['total_illiterates'].sum())
        male_total = int(df['male'].sum())
        female_total = int(df['female'].sum())
        
        top_wards = (
            df[['wardname', 'total_illiterates_by_ward']]
            .drop_duplicates()
            .nlargest(3, 'total_illiterates_by_ward')
        )
        
        context = f"""
        Dataset Analysis Context:
        - Total illiterates: {total_illiterates:,}
        - Male illiterates: {male_total:,}
        - Female illiterates: {female_total:,}
        - Top 3 wards: {', '.join(top_wards['wardname'].tolist())}
        - Ward illiteracy counts: {top_wards['total_illiterates_by_ward'].tolist()}
        
        This is illiteracy data from Rangareddy Urban Area covering 42 wards.
        """
        
        prompt = f"""
        You are a policy analyst specializing in governance and education. Analyze this illiteracy dataset and provide actionable insights.
        
        {context}
        
        Generate a concise AI summary (max 200 words) that includes:
        1. One key strategic insight about the data patterns
        2. One specific policy recommendation with rationale
        3. One implementation priority for immediate action
        
        Focus on practical governance applications. Use professional policy language.
        """
        
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are an expert policy analyst focused on data-driven governance recommendations."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        ai_content = response.choices[0].message.content.strip()
        
        ai_summary = f"""
## AI-Generated Summary

{ai_content}

*Generated using Groq AI (Llama3-8B) for enhanced policy analysis.*
"""
        return ai_summary
        
    except Exception as e:
        print(f"Warning: Groq API call failed: {e}")
        return generate_ai_summary_placeholder(report_content)

def generate_ai_summary_placeholder(report_content):
    """Fallback placeholder function for AI summary."""
    placeholder_summary = """
## AI-Generated Summary

**Key Insight:** The data reveals significant geographic concentration of illiteracy, with the top 3 wards accounting for a disproportionate share of the total illiterate population. This suggests that targeted interventions in these specific areas could yield maximum impact.

**Gender Analysis:** The gender disparity in illiteracy rates indicates the need for tailored approaches that address the specific barriers faced by the more affected gender group.

**Policy Priority:** Immediate focus should be placed on the highest-burden wards while developing scalable solutions that can be replicated across similar demographic areas.

*Note: This summary was generated using fallback analysis. Groq API integration available with valid API key.*
"""
    return placeholder_summary

def insights(args):
    """
    Generate insights report from a transformed CSV file.
    """
    try:
        print_step("INSIGHTS", "Starting insights generation process")
        
        input_path = Path(args.input_file)
        if not input_path.exists():
            print_error(f"Input file '{args.input_file}' not found.")
            sys.exit(1)

        print_step("LOAD", f"Reading transformed file: {input_path.name}")
        df = pd.read_csv(input_path)
        print_success(f"Loaded {len(df)} rows for analysis")

        print_step("ANALYZE", "Generating insights and policy recommendations")
        agent = InsightsAgent()
        report = agent.generate_insights_report(df)
        print_success("Insights report generated")

        # Generate AI-enhanced summary
        print_step("AI_ENHANCE", "Generating AI-powered summary")
        ai_summary = generate_ai_summary_with_groq(report, df)
        enhanced_report = report + ai_summary
        print_success("AI summary added")

        # Prepare output path
        report_path = Path("outputs/insights_report.md")

        # Save output
        print_step("SAVE", "Saving insights report")
        Path("outputs").mkdir(exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(enhanced_report)

        print_success("Successfully generated insights report")
        print(f"   Insights report saved: {report_path}")

        # --- Quick console summary for demo ---
        try:
            print("\nQuick Insights Summary:")
            
            # Top 3 wards
            ward_summary = df[['wardname', 'total_illiterates_by_ward']].drop_duplicates().nlargest(3, 'total_illiterates_by_ward')
            top_names = ", ".join(ward_summary["wardname"].astype(str).tolist())

            # Gender gap
            male_total = df["male"].sum()
            female_total = df["female"].sum()

            if female_total > male_total and male_total > 0:
                gender_gap = round((female_total - male_total) / male_total * 100, 1)
                print(f"   [→] Key finding: Female illiteracy is {gender_gap}% higher than male overall.")
            elif male_total > female_total and female_total > 0:
                gender_gap = round((male_total - female_total) / female_total * 100, 1)
                print(f"   [→] Key finding: Male illiteracy is {gender_gap}% higher than female overall.")
            else:
                 print(f"   [→] Key finding: No significant gender gap in illiteracy rates.")

            print(f"   [→] Recommendation: Target literacy programs in wards: {top_names}.")
        except Exception as e:
            print_error(f"Could not generate quick summary: {e}")

    except Exception as e:
        print_error(f"Error generating insights: {e}")
        sys.exit(1)

def cluster_command(args):
    """Run clustering analysis on multiple CSV files (simplified version)."""
    print_step("CLUSTER", "Starting clustering analysis")
    print("   ⚠️  CrewAI clustering temporarily disabled due to dependency conflicts")
    print("   📄 Use simplified clustering implementation instead")
    
    # Create placeholder output
    results = {
        "status": "disabled",
        "message": "CrewAI clustering temporarily disabled",
        "files_processed": len(args.csv_files) if args.csv_files else 0
    }
    
    output_file = "outputs/clusters.json"
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_success(f"Clustering placeholder created: {output_file}")

def anomalies_command(args):
    """Run anomaly detection on multiple CSV files (simplified version)."""
    print_step("ANOMALIES", "Starting anomaly detection")
    print("   ⚠️  CrewAI anomaly detection temporarily disabled due to dependency conflicts")
    print("   📄 Use simplified anomaly detection implementation instead")
    
    # Create placeholder output
    results = {
        "status": "disabled",
        "message": "CrewAI anomaly detection temporarily disabled",
        "files_processed": len(args.csv_files) if args.csv_files else 0
    }
    
    output_file = "outputs/anomalies.json"
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_success(f"Anomaly detection placeholder created: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="RTGS CLI - Data Processing Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and process a CSV file.")
    ingest_parser.add_argument("input_file", help="Path to the input CSV file.")
    ingest_parser.add_argument("-o", "--output-dir", default="data/cleaned", help="Directory to save the cleaned output files.")
    ingest_parser.set_defaults(func=ingest)

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean a standardized CSV file.")
    clean_parser.add_argument("input_file", help="Path to the standardized CSV file.")
    clean_parser.add_argument("-o", "--output-dir", default="data/cleaned", help="Directory to save the cleaned output file.")
    clean_parser.set_defaults(func=clean)

    # Transform command
    transform_parser = subparsers.add_parser("transform", help="Transform a cleaned CSV file.")
    transform_parser.add_argument("input_file", help="Path to the cleaned CSV file.")
    transform_parser.add_argument("-o", "--output-dir", default="data/cleaned", help="Directory to save the transformed output file.")
    transform_parser.set_defaults(func=transform)

    # Insights command
    insights_parser = subparsers.add_parser("insights", help="Generate insights report from a transformed CSV file.")
    insights_parser.add_argument("input_file", help="Path to the transformed CSV file.")
    insights_parser.set_defaults(func=insights)

    # Cluster command
    cluster_parser = subparsers.add_parser("cluster", help="Cluster wards across multiple CSV files using Hugging Face ML.")
    cluster_parser.add_argument("csv_files", nargs="+", help="Paths to CSV files for clustering analysis.")
    cluster_parser.set_defaults(func=cluster_command)

    # Anomalies command
    anomalies_parser = subparsers.add_parser("anomalies", help="Detect anomalous wards using Hugging Face ML.")
    anomalies_parser.add_argument("csv_files", nargs="+", help="Paths to CSV files for anomaly detection.")
    anomalies_parser.set_defaults(func=anomalies_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
