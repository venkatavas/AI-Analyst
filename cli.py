"""
RTGS CLI - A command-line interface for data processing pipeline.
"""
import json
import typer
from pathlib import Path
import pandas as pd
from typing import Optional

# Import agents
from agents.insights_agent import InsightsAgent

app = typer.Typer(help="RTGS CLI - Data Processing Pipeline")

# Ensure output directories exist
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/cleaned").mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores."""
    df.columns = (
        df.columns.str.lower()
        .str.strip()
        .str.replace(r'[^\w\s]', '_', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )
    return df

@app.command()
def hello(name: str = "World"):
    """
    Say hello to NAME.
    """
    typer.echo(f"Hello, {name}!")

@app.command()
def ingest(
    input_file: str = typer.Argument(
        ...,
        help="Path to the input CSV file in data/raw/ directory"
    ),
    output_dir: str = typer.Option(
        "data/cleaned",
        "--output-dir", "-o",
        help="Directory to save the cleaned output files"
    )
):
    """
    Ingest and process a CSV file, then save standardized version and schema.
    """
    try:
        # Resolve input file path
        input_path = Path(input_file)
        if not input_path.exists():
            typer.echo(f"Error: Input file '{input_file}' not found.", err=True)
            raise typer.Exit(1)
        
        # Read the input file
        typer.echo(f"Reading input file: {input_path}")
        df = pd.read_csv(input_path)
        
        # Generate schema summary
        insights = InsightsAgent()
        schema = insights.generate_schema_summary(df)
        
        # Print schema summary
        typer.echo("\nSchema Summary:")
        typer.echo(f"Total Rows: {schema['total_rows']}")
        typer.echo(f"Total Columns: {schema['total_columns']}")
        typer.echo("\nColumn Details:")
        for col in schema["columns"]:
            typer.echo(f"- {col['name']} ({col['dtype']}): {col['non_null_count']} non-null, {col['unique_count']} unique values")
        
        # Standardize column names
        df_cleaned = standardize_columns(df)
        
        # Prepare output file names
        output_path = Path(output_dir) / f"{input_path.stem}_standardized.csv"
        schema_path = Path("outputs/schema_summary.json")
        
        # Save outputs
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(output_path, index=False)
        with open(schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        typer.echo(f"\n✅ Successfully processed {input_path.name}")
        typer.echo(f"- Cleaned data saved to: {output_path}")
        typer.echo(f"- Schema summary saved to: {schema_path}")
        
    except Exception as e:
        typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
