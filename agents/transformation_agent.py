"""
Transformation Agent - Handles data transformation and feature engineering.
"""
import pandas as pd

class TransformationAgent:
    def transform_data(self, df: pd.DataFrame, file_name: str) -> (pd.DataFrame, dict):
        """Transform the input DataFrame with necessary feature engineering."""
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
        print("   📊 Visualization skipped (disabled for compatibility)")

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
