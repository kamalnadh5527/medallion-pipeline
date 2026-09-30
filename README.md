 # Medallion Pipeline

This project turns uploaded retail CSV files into analytics-ready Gold Parquet tables and a human-readable HTML report. It profiles the source data, generates Source-to-Target Mapping (STTM) rules, and executes approved transformations through Bronze, Silver, and Gold layers. Streamlit provides human review gates, while the CLI supports automated end-to-end runs.

## Architecture

The pipeline has six specialist agents coordinated by a Supervisor:

- **Profiler** analyzes raw CSV structure, types, samples, and possible join keys.
- **STTM Generator** creates Bronze, Silver, and Gold mapping rules.
- **Bronze Agent** performs raw-but-typed CSV ingestion to Parquet.
- **Silver Agent** applies approved cleansing, normalization, deduplication, and typing.
- **Gold Agent** joins Silver tables and materializes analytics aggregations.
- **Reporter** queries Gold data with DuckDB and creates Plotly charts and an HTML report.

The Supervisor coordinates the phases and records audit logs and traces. Human-in-the-loop (HITL) approval pauses the pipeline after each Bronze, Silver, and Gold STTM so mappings can be reviewed before execution.

## Setup

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `ANTHROPIC_API_KEY`. Anthropic is the default provider; the `.env.example` file documents other supported providers.

Run the tests with:

```powershell
pytest
```

## Sample Data

Generate related, intentionally messy retail CSVs in `data/landing/`:

```powershell
python tools/generate_sample_data.py
```

This creates `products.csv`, `stores.csv`, and `sales.csv` with shared keys, nulls, duplicates, inconsistent dates, inconsistent text, and mixed value types.

## Run The App

Start the Streamlit interface:

```powershell
streamlit run app/streamlit_app.py
```

Upload one or more CSVs, enter a business question, review each generated STTM, and approve each layer to continue.

## Run The CLI

The CLI auto-approves STTMs for non-interactive testing:

```powershell
python cli.py --files data/landing/products.csv data/landing/stores.csv data/landing/sales.csv --intent "What are total sales by store and product category?"
```

The CLI prints the profile, STTM, Parquet, and final report paths as each phase completes.
