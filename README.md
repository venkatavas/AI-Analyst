# RTGS CLI

A command-line interface for data processing and analysis pipeline using CrewAI agents.

## Project Structure

```
rtgs-cli/
  ├── cli.py              # Typer-based CLI entrypoint
  ├── agents/             # Folder for CrewAI agents
  │    ├── ingestion_agent.py
  │    ├── cleaning_agent.py
  │    ├── transformation_agent.py
  │    └── insights_agent.py
  ├── data/
  │    ├── raw/           # raw input datasets
  │    └── cleaned/       # standardized + cleaned CSV outputs
  ├── outputs/            # JSON/Markdown reports
  ├── requirements.txt
  └── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rtgs-cli
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Ingest Command

Process a CSV file, standardize column names, and generate a schema summary:

```bash
python cli.py ingest data/raw/your_file.csv
```

Optional arguments:
- `--output-dir`, `-o`: Directory to save the cleaned output files (default: data/cleaned)

Example:
```bash
python cli.py ingest data/raw/Illiterate_Rangareddy_Urban_Area.csv
```

## Development

### Adding New Commands

1. Add a new function in `cli.py` with the `@app.command()` decorator
2. Implement the command logic
3. Add help text using docstrings and parameter descriptions

### Adding New Agents

1. Create a new Python file in the `agents/` directory
2. Define a class with the agent's functionality
3. Import and use the agent in `cli.py` as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.
