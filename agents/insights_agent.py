"""
Insights Agent - Generates insights and reports from the processed data.
"""

import json
from typing import Dict, Any
import pandas as pd

class InsightsAgent:
    def generate_insights_report(self, df: pd.DataFrame) -> str:
        """Generate a Markdown report from the transformed DataFrame."""
        
        # Check if this is governance data or general data
        if self._is_governance_dataset(df):
            return self._generate_governance_insights(df)
        else:
            return self._generate_general_insights(df)
    
    def _is_governance_dataset(self, df: pd.DataFrame) -> bool:
        """Check if dataset has governance structure (ward-based)."""
        ward_cols = ['wardname', 'gp', 'ward_name', 'ward']
        demographic_cols = ['male', 'female', 'total_illiterates']
        
        has_ward = any(col in df.columns for col in ward_cols)
        has_demographics = any(col in df.columns for col in demographic_cols)
        
        return has_ward and has_demographics
    
    def _generate_governance_insights(self, df: pd.DataFrame) -> str:
        """Generate insights for governance datasets."""
        # 1. Extract key metrics
        total_illiterates = int(df['total_illiterates'].sum())
        male_illiterates = int(df['male'].sum())
        female_illiterates = int(df['female'].sum())

        top_wards = (
            df[['wardname', 'total_illiterates_by_ward']]
            .drop_duplicates()
            .nlargest(3, 'total_illiterates_by_ward')
        )

        # 2. Identify trends
        gender_trend = "higher among females" if female_illiterates > male_illiterates else "higher among males"
        concentration_trend = f"heavily concentrated in the top {len(top_wards)} wards."

        # Get geographic context
        geo_info = ""
        if 'muncipality' in df.columns:
            geo_info = f"{df['muncipality'].nunique()} municipalities and "
        elif 'municipality' in df.columns:
            geo_info = f"{df['municipality'].nunique()} municipalities and "
        elif 'mandal' in df.columns:
            geo_info = f"{df['mandal'].nunique()} mandals and "
        
        # 3. Build Markdown report
        report = f"""# Illiteracy Insights Report

## 1. Executive Summary

This report analyzes the provided illiteracy data to identify key trends and provide actionable policy recommendations. The dataset covers {geo_info}{df['wardname'].nunique()} wards.

## 2. Key Findings

- **Total Illiterates:** There are a total of **{total_illiterates}** illiterates in the dataset.
- **Gender Disparity:** Illiteracy is {gender_trend}, with **{female_illiterates}** female illiterates compared to **{male_illiterates}** male illiterates.
- **Geographic Concentration:** Illiteracy is {concentration_trend}

### Top 3 Wards by Illiteracy:

| Rank | Ward Name | Total Illiterates |
|------|-----------|-------------------|
"""

        for i, row in top_wards.iterrows():
            report += f"| {i+1}    | {row['wardname']} | {int(row['total_illiterates_by_ward'])}             |\n"

        report += """
## 3. Policy Recommendations

Based on the analysis, we recommend the following actions:

1.  **Targeted Intervention in High-Prevalence Wards:** Focus literacy programs and resources on the top 3 wards identified, as they represent the most significant concentration of illiteracy.
2.  **Gender-Specific Programs:** Develop and promote literacy initiatives specifically designed to address the gender disparity observed. If female illiteracy is higher, programs could include flexible timings and childcare support.
3.  **Community-Based Awareness Campaigns:** Launch awareness campaigns in the most affected areas to encourage participation in literacy programs and highlight the benefits of education.

"""
        return report
    
    def _generate_general_insights(self, df: pd.DataFrame) -> str:
        """Generate insights for general datasets (non-governance)."""
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Basic statistics
        total_records = len(df)
        
        report = f"""# General Dataset Insights Report

## 1. Executive Summary

This report analyzes a general dataset with {total_records} records and {len(df.columns)} columns.

## 2. Dataset Overview

- **Total Records:** {total_records}
- **Columns:** {len(df.columns)}
- **Numeric Columns:** {len(numeric_cols)}

### Column Analysis:

| Column | Type | Records |
|--------|------|---------|
"""
        
        for col in df.columns:
            col_type = str(df[col].dtype)
            non_null = df[col].count()
            report += f"| {col} | {col_type} | {non_null} |\n"
        
        # Add numeric analysis if available
        if numeric_cols:
            report += f"""

## 3. Numeric Analysis

"""
            for col in numeric_cols:
                if df[col].sum() > 0:  # Only analyze columns with data
                    mean_val = df[col].mean()
                    max_val = df[col].max()
                    min_val = df[col].min()
                    
                    report += f"""### {col}
- **Mean:** {mean_val:.2f}
- **Range:** {min_val} to {max_val}
- **Total:** {df[col].sum()}

"""
        
        report += """## 4. Recommendations

Based on the dataset structure:

1. **Data Quality:** Ensure all numeric fields are properly validated
2. **Analysis Scope:** Consider time-series analysis if temporal data is available
3. **Reporting:** Regular monitoring of key metrics for trend analysis

"""
        return report

    def generate_schema_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a schema summary from the DataFrame.
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Dict containing schema information
        """
        # Check if we have municipality/mandal info for geographic context
        geographic_cols = ['muncipality', 'municipality', 'mandal', 'district']
        available_geo_cols = [col for col in geographic_cols if col in df.columns]
        schema = {
            "columns": [],
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "geographic_context": available_geo_cols
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique())
            }
            schema["columns"].append(col_info)
            
        return schema
