"""
Transformation Agent - Handles data transformation and feature engineering.
"""
import pandas as pd

class TransformationAgent:
    def transform_data(self, df: pd.DataFrame, file_name: str) -> (pd.DataFrame, dict):
        """Transform the input DataFrame with necessary feature engineering."""
        
        # Detect dataset type and handle accordingly
        if self._is_governance_dataset(df):
            return self._transform_governance_data(df, file_name)
        else:
            return self._transform_general_data(df, file_name)
    
    def _is_governance_dataset(self, df: pd.DataFrame) -> bool:
        """Check if dataset has governance structure (ward-based)."""
        ward_cols = ['wardname', 'gp', 'ward_name', 'ward']
        demographic_cols = ['male', 'female']
        
        has_ward = any(col in df.columns for col in ward_cols)
        has_demographics = any(col in df.columns for col in demographic_cols)
        
        return has_ward and has_demographics
    
    def _transform_governance_data(self, df: pd.DataFrame, file_name: str) -> (pd.DataFrame, dict):
        """Transform governance datasets with ward-level data."""
        # Handle different ward name columns
        ward_col = None
        for col in ['wardname', 'gp', 'ward_name', 'ward']:
            if col in df.columns:
                ward_col = col
                break
        
        if ward_col is None:
            raise ValueError("No ward identifier column found (expected: wardname, gp, ward_name, or ward)")
        
        # Rename to standardized column name
        if ward_col != 'wardname':
            df = df.rename(columns={ward_col: 'wardname'})
        
        # Ensure required columns exist
        required_cols = ['wardname', 'male', 'female']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        df[['male', 'female']] = df[['male', 'female']].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        return self._complete_governance_transformation(df, file_name)
    
    def _transform_general_data(self, df: pd.DataFrame, file_name: str) -> (pd.DataFrame, dict):
        """Transform non-governance datasets (like Skill Development)."""
        print(f"[i] Processing non-governance dataset: {file_name}")
        
        # For general datasets, create basic transformations
        # Convert all numeric columns to proper numeric types
        for col in df.columns:
            if col not in ['Activity']:  # Skip text columns
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create summary statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        transformation_report = {
            'dataset_type': 'general',
            'original_columns': list(df.columns),
            'numeric_columns': list(numeric_cols),
            'total_records': len(df),
            'transformations_applied': [
                'Numeric type conversion',
                'Missing value handling'
            ]
        }
        
        return df, transformation_report
    
    def _complete_governance_transformation(self, df: pd.DataFrame, file_name: str) -> (pd.DataFrame, dict):

        # 1. Aggregate total illiterates per ward
        df['total_illiterates'] = df['male'] + df['female']
        df['total_illiterates_by_ward'] = df.groupby('wardname')['total_illiterates'].transform('sum')

        # 2. Compute percentage of male/female illiterates per ward
        # Avoid division by zero
        df['percentage_male'] = (df['male'] / df['total_illiterates_by_ward'] * 100).fillna(0)
        df['percentage_female'] = (df['female'] / df['total_illiterates_by_ward'] * 100).fillna(0)

        # 3. Rank wards by total illiterates
        df['rank_by_illiterates'] = df['total_illiterates_by_ward'].rank(method='dense', ascending=False).astype(int)

        # Skip visualization to avoid rendering issues
        print("   Visualization skipped (disabled for compatibility)")

        # 4. Generate JSON report
        total_illiterates_overall = int(df['total_illiterates'].sum())
        agg_by_gender = {
            'male': int(df['male'].sum()),
            'female': int(df['female'].sum())
        }
        top_wards_df = df.groupby('wardname')['total_illiterates'].sum().nlargest(5)
        top_wards = [
            {"ward": ward, "illiterates": int(count)}
            for ward, count in top_wards_df.items()
        ]

        report = {
            "dataset_name": file_name,
            "total_illiterates": total_illiterates_overall,
            "aggregations": {
                "by_gender": agg_by_gender,
                "top_wards": top_wards
            },
            "transformed_columns": [
                'total_illiterates_by_ward',
                'percentage_male',
                'percentage_female',
                'rank_by_illiterates'
            ],
            "notes": f"Dataset transformed successfully. Found {total_illiterates_overall} total illiterates across {df['wardname'].nunique()} wards."
        }

        return df, report
