#!/usr/bin/env python3
"""
RTGS CLI MCP Server

Exposes the Real-Time Governance System data processing pipeline 
through the Model Context Protocol (MCP) for AI system integration.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Import our CLI functionality
from cli import (
    standardize_columns,
    generate_schema_csv,
    generate_visualization,
    generate_ai_summary_with_groq
)
from agents.ingestion_agent import IngestionAgent
from agents.cleaning_agent import CleaningAgent
from agents.transformation_agent import TransformationAgent
from agents.insights_agent import InsightsAgent
from agents.ml_agent import MLAgent

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the MCP server
server = Server("rtgs-cli")

@server.list_resources()
async def handle_list_resources() -> list[Resource]:
    """List available RTGS CLI resources."""
    resources = []
    
    # Check for existing outputs
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        for file_path in outputs_dir.glob("*"):
            if file_path.is_file():
                resources.append(Resource(
                    uri=f"rtgs://outputs/{file_path.name}",
                    name=f"RTGS Output: {file_path.name}",
                    description=f"Generated output file: {file_path.name}",
                    mimeType=_get_mime_type(file_path)
                ))
    
    # Check for processed data files
    data_dir = Path("data/cleaned")
    if data_dir.exists():
        for file_path in data_dir.glob("*.csv"):
            resources.append(Resource(
                uri=f"rtgs://data/{file_path.name}",
                name=f"RTGS Data: {file_path.name}",
                description=f"Processed dataset: {file_path.name}",
                mimeType="text/csv"
            ))
    
    # Add pipeline status resource
    resources.append(Resource(
        uri="rtgs://status/pipeline",
        name="RTGS Pipeline Status",
        description="Current status of the data processing pipeline",
        mimeType="application/json"
    ))
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read RTGS CLI resource content."""
    if uri.startswith("rtgs://outputs/"):
        filename = uri.replace("rtgs://outputs/", "")
        file_path = Path("outputs") / filename
        
        if file_path.exists():
            if file_path.suffix == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.suffix == ".md":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            elif file_path.suffix == ".csv":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                return f"Binary file: {filename}"
        else:
            raise FileNotFoundError(f"Output file not found: {filename}")
    
    elif uri.startswith("rtgs://data/"):
        filename = uri.replace("rtgs://data/", "")
        file_path = Path("data/cleaned") / filename
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"Data file not found: {filename}")
    
    elif uri == "rtgs://status/pipeline":
        return _get_pipeline_status()
    
    else:
        raise ValueError(f"Unknown resource URI: {uri}")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List available RTGS CLI tools."""
    return [
        Tool(
            name="rtgs_ingest",
            description="Ingest and standardize a CSV dataset",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the CSV file to ingest"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save outputs (default: data/cleaned)",
                        "default": "data/cleaned"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="rtgs_clean",
            description="Clean a standardized dataset (remove duplicates, handle missing values)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the standardized CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="rtgs_transform",
            description="Transform cleaned data (add aggregations, rankings, percentages)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the cleaned CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="rtgs_insights",
            description="Generate policy insights and AI-powered analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the transformed CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="rtgs_pipeline",
            description="Run the complete RTGS pipeline (ingest -> clean -> transform -> insights)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the raw CSV file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="rtgs_query_data",
            description="Query processed data with natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about the data"
                    },
                    "dataset": {
                        "type": "string",
                        "description": "Dataset to query (default: latest transformed data)",
                        "default": "latest"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="rtgs_cluster",
            description="Cluster wards across multiple CSV files using CrewAI ML",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of CSV file paths for clustering analysis"
                    }
                },
                "required": ["csv_files"]
            }
        ),
        Tool(
            name="rtgs_anomalies",
            description="Detect anomalous wards using CrewAI ML",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of CSV file paths for anomaly detection"
                    }
                },
                "required": ["csv_files"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle RTGS CLI tool calls."""
    try:
        if name == "rtgs_ingest":
            return await _handle_ingest(arguments)
        elif name == "rtgs_clean":
            return await _handle_clean(arguments)
        elif name == "rtgs_transform":
            return await _handle_transform(arguments)
        elif name == "rtgs_insights":
            return await _handle_insights(arguments)
        elif name == "rtgs_pipeline":
            return await _handle_pipeline(arguments)
        elif name == "rtgs_query_data":
            return await _handle_query_data(arguments)
        elif name == "rtgs_cluster":
            return await _handle_cluster(arguments)
        elif name == "rtgs_anomalies":
            return await _handle_anomalies(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing {name}: {str(e)}"
        )]

# Tool implementation functions
async def _handle_ingest(args: dict) -> list[TextContent]:
    """Handle ingest tool call."""
    file_path = Path(args["file_path"])
    output_dir = Path(args.get("output_dir", "data/cleaned"))
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Read and process the CSV
    df = pd.read_csv(file_path)
    df_cleaned = standardize_columns(df)
    
    # Generate schema
    insights = InsightsAgent()
    schema = insights.generate_schema_summary(df_cleaned)
    
    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    output_path = output_dir / f"{file_path.stem}_standardized.csv"
    schema_json_path = Path("outputs/schema_summary.json")
    schema_csv_path = Path("outputs/schema_summary.csv")
    
    df_cleaned.to_csv(output_path, index=False)
    with open(schema_json_path, 'w') as f:
        json.dump(schema, f, indent=2)
    generate_schema_csv(df_cleaned, schema, schema_csv_path)
    
    return [TextContent(
        type="text",
        text=f"Successfully ingested {file_path.name}\n"
             f"- Standardized data: {output_path}\n"
             f"- Schema JSON: {schema_json_path}\n"
             f"- Schema CSV: {schema_csv_path}\n"
             f"- Rows: {len(df_cleaned):,}\n"
             f"- Columns: {len(df_cleaned.columns)}"
    )]

async def _handle_clean(args: dict) -> list[TextContent]:
    """Handle clean tool call."""
    file_path = Path(args["file_path"])
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Apply cleaning
    agent = CleaningAgent()
    df_cleaned, report = agent.clean_data(df)
    
    # Save outputs
    output_path = Path("data/cleaned") / f"{file_path.stem}_cleaned.csv"
    report_path = Path("outputs/cleaning_report.json")
    
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    df_cleaned.to_csv(output_path, index=False)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return [TextContent(
        type="text",
        text=f"Successfully cleaned {file_path.name}\n"
             f"- Cleaned data: {output_path}\n"
             f"- Cleaning report: {report_path}\n"
             f"- Duplicates removed: {report.get('duplicates_removed', 0)}\n"
             f"- Missing values filled: {sum(report.get('missing_values_filled', {}).values())}\n"
             f"- Outliers replaced: {sum(report.get('outliers_replaced', {}).values())}"
    )]

async def _handle_transform(args: dict) -> list[TextContent]:
    """Handle transform tool call."""
    file_path = Path(args["file_path"])
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Apply transformations
    agent = TransformationAgent()
    df_transformed, report = agent.transform_data(df)
    
    # Save outputs
    output_path = Path("data/cleaned") / f"{file_path.stem.replace('_cleaned', '')}_transformed.csv"
    report_path = Path("outputs/transformation_report.json")
    viz_path = Path("outputs/trend.png")
    
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    df_transformed.to_csv(output_path, index=False)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate visualization
    viz_success = generate_visualization(df_transformed, viz_path)
    
    result_text = (f"Successfully transformed {file_path.name}\n"
                  f"- Transformed data: {output_path}\n"
                  f"- Transformation report: {report_path}")
    
    if viz_success:
        result_text += f"\n- Visualization: {viz_path}"
    
    return [TextContent(type="text", text=result_text)]

async def _handle_insights(args: dict) -> list[TextContent]:
    """Handle insights tool call."""
    file_path = Path(args["file_path"])
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Generate insights
    agent = InsightsAgent()
    report = agent.generate_insights_report(df)
    
    # Add AI summary
    ai_summary = generate_ai_summary_with_groq(report, df)
    enhanced_report = report + ai_summary
    
    # Save output
    report_path = Path("outputs/insights_report.md")
    Path("outputs").mkdir(exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(enhanced_report)
    
    # Generate quick summary
    total_illiterates = int(df['total_illiterates'].sum()) if 'total_illiterates' in df.columns else 0
    male_total = int(df['male'].sum()) if 'male' in df.columns else 0
    female_total = int(df['female'].sum()) if 'female' in df.columns else 0
    
    summary_text = f"Successfully generated insights for {file_path.name}\n"
    summary_text += f"- Insights report: {report_path}\n"
    summary_text += f"- Total illiterates: {total_illiterates:,}\n"
    
    if male_total > 0 and female_total > 0:
        if female_total > male_total:
            gap_pct = ((female_total - male_total) / male_total) * 100
            summary_text += f"- Gender gap: Female illiteracy {gap_pct:.1f}% higher than male\n"
        else:
            gap_pct = ((male_total - female_total) / female_total) * 100
            summary_text += f"- Gender gap: Male illiteracy {gap_pct:.1f}% higher than female\n"
    
    return [TextContent(type="text", text=summary_text)]

async def _handle_pipeline(args: dict) -> list[TextContent]:
    """Handle complete pipeline execution."""
    file_path = Path(args["file_path"])
    
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    results = []
    
    # Step 1: Ingest
    ingest_result = await _handle_ingest({"file_path": str(file_path)})
    results.extend(ingest_result)
    
    # Step 2: Clean
    standardized_path = Path("data/cleaned") / f"{file_path.stem}_standardized.csv"
    clean_result = await _handle_clean({"file_path": str(standardized_path)})
    results.extend(clean_result)
    
    # Step 3: Transform
    cleaned_path = Path("data/cleaned") / f"{file_path.stem}_standardized_cleaned.csv"
    transform_result = await _handle_transform({"file_path": str(cleaned_path)})
    results.extend(transform_result)
    
    # Step 4: Insights
    transformed_path = Path("data/cleaned") / f"{file_path.stem}_transformed.csv"
    insights_result = await _handle_insights({"file_path": str(transformed_path)})
    results.extend(insights_result)
    
    # Combine all results
    combined_text = "RTGS Pipeline Complete!\n\n" + "\n\n".join([r.text for r in results])
    
    return [TextContent(type="text", text=combined_text)]

async def _handle_query_data(args: dict) -> list[TextContent]:
    """Handle natural language data queries."""
    query = args["query"]
    dataset = args.get("dataset", "latest")
    
    # Find the latest transformed dataset
    data_dir = Path("data/cleaned")
    transformed_files = list(data_dir.glob("*_transformed.csv"))
    
    if not transformed_files:
        return [TextContent(
            type="text",
            text="No transformed datasets found. Please run the pipeline first."
        )]
    
    # Use the most recent file
    latest_file = max(transformed_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    
    # Simple query processing (can be enhanced with NLP)
    response = _process_data_query(df, query)
    
    return [TextContent(
        type="text",
        text=f"Query: {query}\n\nDataset: {latest_file.name}\n\nResponse:\n{response}"
    )]

# Helper functions
def _get_mime_type(file_path: Path) -> str:
    """Get MIME type for file."""
    suffix = file_path.suffix.lower()
    mime_types = {
        '.json': 'application/json',
        '.csv': 'text/csv',
        '.md': 'text/markdown',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg'
    }
    return mime_types.get(suffix, 'application/octet-stream')

def _get_pipeline_status() -> str:
    """Get current pipeline status."""
    status = {
        "pipeline_version": "1.0.0",
        "available_outputs": [],
        "available_datasets": [],
        "last_run": None
    }
    
    # Check for outputs
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        status["available_outputs"] = [f.name for f in outputs_dir.glob("*") if f.is_file()]
    
    # Check for datasets
    data_dir = Path("data/cleaned")
    if data_dir.exists():
        status["available_datasets"] = [f.name for f in data_dir.glob("*.csv")]
    
    # Check for recent insights report
    insights_file = Path("outputs/insights_report.md")
    if insights_file.exists():
        status["last_run"] = insights_file.stat().st_mtime
    
    return json.dumps(status, indent=2)

def _process_data_query(df: pd.DataFrame, query: str) -> str:
    """Process natural language queries about the data."""
    query_lower = query.lower()
    
    # Basic query patterns
    if "total" in query_lower and "illiterate" in query_lower:
        if 'total_illiterates' in df.columns:
            total = int(df['total_illiterates'].sum())
            return f"Total illiterates in the dataset: {total:,}"
    
    elif "gender" in query_lower or "male" in query_lower or "female" in query_lower:
        if 'male' in df.columns and 'female' in df.columns:
            male_total = int(df['male'].sum())
            female_total = int(df['female'].sum())
            return f"Gender breakdown:\n- Male: {male_total:,}\n- Female: {female_total:,}"
    
    elif "top" in query_lower and "ward" in query_lower:
        if 'wardname' in df.columns and 'total_illiterates_by_ward' in df.columns:
            top_wards = (
                df[['wardname', 'total_illiterates_by_ward']]
                .drop_duplicates()
                .nlargest(5, 'total_illiterates_by_ward')
            )
            result = "Top 5 wards by illiteracy:\n"
            for _, row in top_wards.iterrows():
                result += f"- {row['wardname']}: {int(row['total_illiterates_by_ward']):,}\n"
            return result
    
    elif "column" in query_lower or "field" in query_lower:
        return f"Available columns: {', '.join(df.columns.tolist())}"
    
    elif "row" in query_lower or "record" in query_lower:
        return f"Dataset contains {len(df):,} rows"
    
    else:
        return f"I can help you query the data. Try asking about:\n" \
               f"- Total illiterates\n" \
               f"- Gender breakdown\n" \
               f"- Top wards\n" \
               f"- Available columns\n" \
               f"- Number of records"

async def _handle_cluster(args: dict) -> list[TextContent]:
    """Handle clustering tool call."""
    csv_files = args["csv_files"]
    
    # Validate files exist
    valid_files = []
    for csv_file in csv_files:
        file_path = Path(csv_file)
        if file_path.exists():
            valid_files.append(str(file_path))
        else:
            return [TextContent(type="text", text=f"File not found: {csv_file}")]
    
    if not valid_files:
        return [TextContent(type="text", text="No valid CSV files provided for clustering")]
    
    # Initialize ML Agent and perform clustering
    ml_agent = MLAgent()
    results = ml_agent.cluster_wards(valid_files)
    
    if "error" in results:
        return [TextContent(type="text", text=f"Clustering failed: {results['error']}")]
    
    # Save results
    output_path = Path("outputs/clusters.json")
    Path("outputs").mkdir(exist_ok=True)
    
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    clustering_results = results.get("clustering_results", {})
    summary_text = f"Ward clustering analysis completed\n"
    summary_text += f"- Files analyzed: {len(valid_files)}\n"
    summary_text += f"- Total wards: {clustering_results.get('total_wards', 0)}\n"
    summary_text += f"- Clusters found: {clustering_results.get('num_clusters', 0)}\n"
    summary_text += f"- Silhouette score: {clustering_results.get('silhouette_score', 'N/A')}\n"
    summary_text += f"- Results saved: {output_path}\n"
    summary_text += f"- AI Framework: {results.get('methodology', {}).get('ai_framework', 'CrewAI')}"
    
    return [TextContent(type="text", text=summary_text)]

async def _handle_anomalies(args: dict) -> list[TextContent]:
    """Handle anomaly detection tool call."""
    csv_files = args["csv_files"]
    
    # Validate files exist
    valid_files = []
    for csv_file in csv_files:
        file_path = Path(csv_file)
        if file_path.exists():
            valid_files.append(str(file_path))
        else:
            return [TextContent(type="text", text=f"File not found: {csv_file}")]
    
    if not valid_files:
        return [TextContent(type="text", text="No valid CSV files provided for anomaly detection")]
    
    # Initialize ML Agent and perform anomaly detection
    ml_agent = MLAgent()
    results = ml_agent.detect_anomalies(valid_files)
    
    if "error" in results:
        return [TextContent(type="text", text=f"Anomaly detection failed: {results['error']}")]
    
    # Save results
    output_path = Path("outputs/anomalies.json")
    Path("outputs").mkdir(exist_ok=True)
    
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Generate summary
    anomaly_results = results.get("anomaly_detection_results", {})
    summary_text = f"Anomaly detection analysis completed\n"
    summary_text += f"- Files analyzed: {len(valid_files)}\n"
    summary_text += f"- Total wards: {anomaly_results.get('total_wards_analyzed', 0)}\n"
    summary_text += f"- Anomalies detected: {anomaly_results.get('anomalies_detected', 0)}\n"
    summary_text += f"- Anomaly rate: {anomaly_results.get('anomaly_percentage', 0)}%\n"
    summary_text += f"- Results saved: {output_path}\n"
    summary_text += f"- AI Framework: {results.get('methodology', {}).get('ai_framework', 'CrewAI')}"
    
    return [TextContent(type="text", text=summary_text)]

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="rtgs-cli",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                )
            )
        )

if __name__ == "__main__":
    asyncio.run(main())
