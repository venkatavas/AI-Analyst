"""
Ingestion Agent - Handles the initial data loading and validation.
"""
import pandas as pd
import json
from pathlib import Path

class IngestionAgent:
    def __init__(self):
        pass

    def ingest_csv(self, file_path: str):
        """Ingest and standardize a CSV file."""
        try:
            input_path = Path(file_path)
            
            # Load CSV
            df = pd.read_csv(input_path)
            
            # Standardize column names (lowercase, underscores)
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
            
            # Generate schema summary
            schema_summary = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'columns': {}
            }
            
            for col in df.columns:
                schema_summary['columns'][col] = {
                    'dtype': str(df[col].dtype),
                    'non_null': int(df[col].count()),
                    'null_count': int(df[col].isnull().sum()),
                    'unique_count': int(df[col].nunique())
                }
            
            # Save standardized file
            output_path = Path("data/raw") / f"{input_path.stem}_standardized.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            
            # Save schema files
            schema_json_path = Path("outputs/schema_summary.json")
            schema_csv_path = Path("outputs/schema_summary.csv")
            schema_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(schema_json_path, 'w') as f:
                json.dump(schema_summary, f, indent=2)
            
            # Create schema CSV
            schema_df = pd.DataFrame([
                {
                    'column': col,
                    'dtype': info['dtype'],
                    'non_null': info['non_null'],
                    'null_count': info['null_count'],
                    'unique_count': info['unique_count']
                }
                for col, info in schema_summary['columns'].items()
            ])
            schema_df.to_csv(schema_csv_path, index=False)
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'schema_summary': f"outputs/schema_summary.json",
                'message': f"Successfully processed {len(df)} rows and {len(df.columns)} columns"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Ingestion failed: {str(e)}"
            }

    def process(self, file_path: str):
        """Process the input file and return a pandas DataFrame."""
        return self.ingest_csv(file_path)
