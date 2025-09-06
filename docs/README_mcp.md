# MCP Server Integration

## Overview

The RTGS CLI includes a Model Context Protocol (MCP) server that exposes the data processing pipeline as resources and tools for AI system integration. This enables other AI applications to access and utilize the governance data processing capabilities.

## MCP Server Features

### Resources
The MCP server exposes the following resources:

- **Output Files**: All generated reports, schemas, and visualizations
- **Processed Data**: Standardized, cleaned, and transformed datasets
- **Pipeline Status**: Real-time status of the processing pipeline

### Tools
Available tools through the MCP interface:

1. **rtgs_ingest** - Ingest and standardize CSV datasets
2. **rtgs_clean** - Clean data (remove duplicates, handle missing values)
3. **rtgs_transform** - Transform data (aggregations, rankings, percentages)
4. **rtgs_insights** - Generate policy insights with AI analysis
5. **rtgs_pipeline** - Run complete pipeline end-to-end
6. **rtgs_query_data** - Natural language queries on processed data

## Setup and Configuration

### 1. Install Dependencies
```bash
pip install mcp>=0.1.0
```

### 2. Start MCP Server
```bash
python mcp_server.py
```

### 3. Configuration File
The `mcp_config.json` file contains server configuration:

```json
{
  "mcpServers": {
    "rtgs-cli": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": ".",
      "env": {
        "GROQ_API_KEY": "${GROQ_API_KEY}"
      }
    }
  }
}
```

## Usage Examples

### Resource Access
```python
# List available resources
resources = await client.list_resources()

# Read insights report
report = await client.read_resource("rtgs://outputs/insights_report.md")

# Get pipeline status
status = await client.read_resource("rtgs://status/pipeline")
```

### Tool Usage
```python
# Run complete pipeline
result = await client.call_tool("rtgs_pipeline", {
    "file_path": "data/raw/dataset.csv"
})

# Query data with natural language
query_result = await client.call_tool("rtgs_query_data", {
    "query": "What are the top 5 wards by illiteracy?",
    "dataset": "latest"
})

# Generate insights only
insights = await client.call_tool("rtgs_insights", {
    "file_path": "data/cleaned/dataset_transformed.csv"
})
```

## Integration Scenarios

### 1. AI Assistant Integration
- **Chatbots**: Enable conversational data analysis
- **Virtual Analysts**: Automated report generation
- **Policy Advisors**: Real-time governance insights

### 2. Workflow Automation
- **Data Pipelines**: Automated processing of new datasets
- **Report Generation**: Scheduled analysis and reporting
- **Alert Systems**: Anomaly detection and notifications

### 3. Dashboard Integration
- **Real-time Updates**: Live data processing status
- **Interactive Queries**: Natural language data exploration
- **Visualization Embedding**: Charts and graphs in web interfaces

## API Reference

### Resources

#### `rtgs://outputs/{filename}`
Access generated output files (JSON, CSV, Markdown, PNG)

#### `rtgs://data/{filename}`
Access processed dataset files

#### `rtgs://status/pipeline`
Get current pipeline status and available resources

### Tools

#### `rtgs_ingest`
**Parameters:**
- `file_path` (required): Path to CSV file
- `output_dir` (optional): Output directory

**Returns:** Processing summary with file locations

#### `rtgs_clean`
**Parameters:**
- `file_path` (required): Path to standardized CSV

**Returns:** Cleaning report with statistics

#### `rtgs_transform`
**Parameters:**
- `file_path` (required): Path to cleaned CSV

**Returns:** Transformation summary with new features

#### `rtgs_insights`
**Parameters:**
- `file_path` (required): Path to transformed CSV

**Returns:** Policy insights and AI analysis

#### `rtgs_pipeline`
**Parameters:**
- `file_path` (required): Path to raw CSV

**Returns:** Complete pipeline execution results

#### `rtgs_query_data`
**Parameters:**
- `query` (required): Natural language query
- `dataset` (optional): Dataset identifier

**Returns:** Query results and analysis

## Natural Language Queries

The MCP server supports natural language queries for data exploration:

### Supported Query Types
- **Total counts**: "How many total illiterates are there?"
- **Gender analysis**: "What's the gender breakdown?"
- **Geographic insights**: "Which are the top wards by illiteracy?"
- **Data structure**: "What columns are available?"
- **Dataset info**: "How many records are there?"

### Example Queries
```
"What are the top 5 wards with highest illiteracy?"
"Show me the gender breakdown of illiterates"
"How many total illiterates are in the dataset?"
"What columns are available in the data?"
```

## Error Handling

The MCP server includes comprehensive error handling:

- **File Not Found**: Clear error messages for missing files
- **Invalid Parameters**: Validation of tool parameters
- **Processing Errors**: Graceful handling of data processing failures
- **API Failures**: Fallback for Groq API issues

## Security Considerations

- **Environment Variables**: API keys stored securely in `.env`
- **File Access**: Restricted to project directories
- **Input Validation**: All parameters validated before processing
- **Error Sanitization**: No sensitive information in error messages

## Performance

- **Async Operations**: Non-blocking tool execution
- **Resource Caching**: Efficient resource access
- **Memory Management**: Optimized for large datasets
- **Concurrent Requests**: Multiple tool calls supported

## Troubleshooting

### Common Issues

1. **MCP Server Won't Start**
   - Check Python path and dependencies
   - Verify mcp package installation
   - Ensure proper working directory

2. **Tool Execution Fails**
   - Verify file paths exist
   - Check data format compatibility
   - Review error messages for specifics

3. **Resource Access Issues**
   - Ensure outputs directory exists
   - Check file permissions
   - Verify resource URI format

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export MCP_LOG_LEVEL=DEBUG
```
