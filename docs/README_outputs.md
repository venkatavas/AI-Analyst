# Output Reference

## Overview

RTGS CLI generates comprehensive outputs organized in the `/outputs/` directory. Each file serves a specific purpose for governance analysis and policy decision-making.

## File Structure

```
outputs/
├── schema_summary.json      # Detailed dataset metadata
├── schema_summary.csv       # Tabular schema information
├── cleaning_report.json     # Data quality improvements log
├── transformation_report.json # Statistical analysis results
├── insights_report.md       # Policy recommendations
└── trend.png               # Visualization of key findings
```

## Detailed File Descriptions

### 1. Schema Summary Files

#### `schema_summary.json`
**Purpose:** Complete dataset metadata for technical analysis

**Sample Content:**
```json
{
  "columns": [
    {
      "name": "district",
      "dtype": "object",
      "non_null_count": 126,
      "null_count": 0,
      "unique_count": 1
    },
    {
      "name": "wardname", 
      "dtype": "object",
      "non_null_count": 126,
      "null_count": 0,
      "unique_count": 42
    },
    {
      "name": "female",
      "dtype": "int64", 
      "non_null_count": 126,
      "null_count": 0,
      "unique_count": 55
    }
  ],
  "total_columns": 7,
  "total_rows": 126
}
```

**Use Cases:**
- Data validation and quality assessment
- Integration planning with other datasets
- Technical documentation for developers

#### `schema_summary.csv`
**Purpose:** Human-readable schema information for analysts

| column_name | data_type | non_null_count | null_count | unique_count | range_or_values | unit_type |
|-------------|-----------|----------------|------------|--------------|-----------------|-----------|
| district    | object    | 126            | 0          | 1            | 1 unique values | categorical |
| wardname    | object    | 126            | 0          | 42           | 42 unique values| categorical |
| female      | int64     | 126            | 0          | 55           | 0 - 89          | count |
| male        | int64     | 126            | 0          | 48           | 0 - 67          | count |

**Use Cases:**
- Quick data profiling
- Spreadsheet analysis
- Stakeholder reporting

### 2. Data Quality Reports

#### `cleaning_report.json`
**Purpose:** Detailed log of all data cleaning operations

**Sample Content:**
```json
{
  "original_rows": 126,
  "final_rows": 126,
  "duplicates_removed": 0,
  "missing_values_filled": {
    "female": 2,
    "male": 1
  },
  "outliers_replaced": {
    "female": 3,
    "male": 2
  },
  "cleaning_operations": [
    "Removed duplicate rows",
    "Filled missing numeric values with column mean",
    "Replaced outliers beyond 3 standard deviations"
  ],
  "data_quality_score": 0.95
}
```

**Use Cases:**
- Data audit trails for compliance
- Quality assurance validation
- Process improvement analysis

### 3. Statistical Analysis

#### `transformation_report.json`
**Purpose:** Comprehensive statistical transformations and aggregations

**Sample Content:**
```json
{
  "total_illiterates": 6789,
  "aggregations": {
    "by_gender": {
      "male": 2567,
      "female": 4222,
      "transgender": 0
    },
    "by_ward": {
      "total_wards": 42,
      "avg_illiterates_per_ward": 161.6
    },
    "top_wards": [
      {
        "ward": "Nadergul",
        "illiterates": 456,
        "percentage_of_total": 6.7
      },
      {
        "ward": "Revenue Ward No 2", 
        "illiterates": 398,
        "percentage_of_total": 5.9
      }
    ]
  },
  "derived_features": [
    "total_illiterates_by_ward",
    "percentage_male",
    "percentage_female", 
    "rank_by_illiterates"
  ]
}
```

**Use Cases:**
- Statistical analysis and modeling
- Performance benchmarking
- Resource allocation planning

### 4. Policy Insights

#### `insights_report.md`
**Purpose:** Executive summary with actionable policy recommendations

**Sample Content:**
```markdown
# Illiteracy Insights Report

## 1. Executive Summary

This report analyzes illiteracy data covering 1 municipality and 42 wards, 
providing actionable policy recommendations for targeted interventions.

## 2. Key Findings

- **Total Illiterates:** 6,789 individuals across all wards
- **Gender Disparity:** Female illiteracy is 75.7% higher than male
- **Geographic Concentration:** Top 3 wards account for 18.5% of total illiteracy

### Top 3 Wards by Illiteracy:

| Rank | Ward Name | Total Illiterates |
|------|-----------|-------------------|
| 1    | Nadergul  | 456              |
| 2    | Revenue Ward No 2 | 398      |
| 3    | Revenue Ward No 8 | 367      |

## 3. Policy Recommendations

1. **Targeted Intervention:** Focus literacy programs on top 3 wards
2. **Gender-Specific Programs:** Address barriers to female education
3. **Community Engagement:** Launch awareness campaigns in high-burden areas

## AI-Generated Summary

**Key Insight:** Geographic concentration suggests targeted interventions 
could yield maximum impact with focused resource allocation.
```

**Use Cases:**
- Executive briefings and presentations
- Policy document preparation
- Stakeholder communication

### 5. Visualizations

#### `trend.png`
**Purpose:** Visual representation of key findings

**Content:** Bar chart showing top 5 wards by illiteracy count with:
- Color-coded bars for visual appeal
- Value labels for precise readings
- Professional formatting for presentations

**Use Cases:**
- Presentation slides and reports
- Dashboard integration
- Public communication materials

## Output Integration Workflows

### For Policymakers
1. **Quick Overview:** Read `insights_report.md` executive summary
2. **Visual Briefing:** Review `trend.png` for key patterns
3. **Deep Dive:** Examine `transformation_report.json` for detailed statistics

### For Data Analysts
1. **Data Validation:** Check `schema_summary.csv` for data quality
2. **Process Audit:** Review `cleaning_report.json` for methodology
3. **Statistical Analysis:** Use `transformation_report.json` for modeling

### For Technical Teams
1. **Integration Planning:** Use `schema_summary.json` for system design
2. **Quality Monitoring:** Track `cleaning_report.json` metrics over time
3. **Pipeline Optimization:** Analyze processing logs for improvements

## Traceability and Accountability

### Audit Trail
Every output file includes:
- **Timestamp:** When the analysis was performed
- **Source Data:** Original dataset identification
- **Processing Steps:** Complete methodology documentation
- **Quality Metrics:** Data reliability indicators

### Reproducibility
The combination of outputs enables:
- **Full reproduction** of analysis results
- **Methodology validation** by independent reviewers
- **Historical comparison** across multiple analysis runs
- **Process improvement** through detailed logging

### Compliance Features
- **Data lineage tracking** through file naming conventions
- **Quality assurance** through comprehensive reporting
- **Change management** via detailed transformation logs
- **Stakeholder transparency** through accessible documentation
