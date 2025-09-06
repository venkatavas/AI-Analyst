"""
Insights Agent - Generates insights and reports from the processed data.
"""

import json
from typing import Dict, Any
import pandas as pd

class InsightsAgent:
    def generate_insights_report(self, df: pd.DataFrame) -> str:
        """Generate a Markdown report from the transformed DataFrame."""
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
