"""
Cleaning Agent - Handles data cleaning and standardization.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

class CleaningAgent:
    def __init__(self):
        pass

    def clean_csv(self, file_path: str):
        """Clean a CSV file and save results."""
        try:
            input_path = Path(file_path)
            
            # Load CSV
            df = pd.read_csv(input_path)
            
            # Clean data
            cleaned_df, report = self.clean_data(df)
            
            # Save cleaned file
            output_path = Path("data/cleaned") / f"{input_path.stem}_cleaned.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_csv(output_path, index=False)
            
            # Save cleaning report
            report_path = Path("outputs/cleaning_report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return {
                'status': 'success',
                'output_file': str(output_path),
                'report_file': str(report_path),
                'message': f"Successfully cleaned {len(cleaned_df)} rows"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Cleaning failed: {str(e)}"
            }

    def clean_data(self, df: pd.DataFrame) -> (pd.DataFrame, dict):
        """Clean and standardize the input DataFrame."""
        report = {
            'duplicates_removed': 0,
            'missing_values_filled': {},
            'outliers_replaced': {}
        }

        # 1. Drop duplicates
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        report['duplicates_removed'] = initial_rows - len(df_cleaned)

        # 2. Handle missing values
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                num_missing = int(df_cleaned[col].isnull().sum())
                report['missing_values_filled'][col] = num_missing
                
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    # Fill numeric with mean
                    mean_val = df_cleaned[col].mean()
                    df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                else:
                    # Fill categorical with 'Unknown'
                    df_cleaned[col] = df_cleaned[col].fillna('Unknown')

        # 3. Detect and replace outliers in numeric columns
        numeric_cols = df_cleaned.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
            if not outliers.empty:
                report['outliers_replaced'][col] = len(outliers)
                median_val = df_cleaned[col].median()
                df_cleaned.loc[outliers.index, col] = median_val

        return df_cleaned, report
