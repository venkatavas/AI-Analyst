"""
Cleaning Agent - Handles data cleaning and standardization.
"""
import pandas as pd
import numpy as np

class CleaningAgent:
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
