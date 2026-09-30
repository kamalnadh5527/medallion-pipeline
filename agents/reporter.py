import json
import re
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import REPORTS_DIR
from core.llm import _make_llm
from core.memory import store_document
from core.observability import AgentTrace


_DUCKDB = duckdb.connect(database=":memory:")
_QUERY_ATTEMPTS: dict[str, int] = {}
_TABLE_NAMES: dict[str, str] = {}


def _table_name(path: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", Path(path).stem).strip("_")
    return f"gold_{stem.lower()}"


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@tool
def inspect_gold_tables_tool(gold_paths: list[str]) -> str:
    """Preview each Gold table schema, row count, and three sample rows."""
    previews = []
    for gold_path in gold_paths:
        frame = pd.read_parquet(gold_path)
        previews.append(
            {
                "file_path": str(gold_path),
                "table_name": _table_name(gold_path),
                "row_count": len(frame),
                "schema": [
                    {"name": str(column), "dtype": str(frame[column].dtype)}
                    for column in frame.columns
                ],
                "sample_rows": frame.head(3).to_dict(orient="records"),
            }
        )
    return json.dumps({"gold_tables": previews}, default=str)


@tool
def load_gold_data_tool(gold_paths: list[str]) -> str:
    """Register Gold Parquet files as DuckDB tables and return their catalog."""
    _TABLE_NAMES.clear()
    catalog = []
    for gold_path in gold_paths:
        table_name = _table_name(gold_path)
        quoted_table = _quote_identifier(table_name)
        _DUCKDB.execute(
            f"CREATE OR REPLACE TABLE {quoted_table} AS "
            "SELECT * FROM read_parquet(?)",
            [str(gold_path)],
        )
        _TABLE_NAMES[str(gold_path)] = table_name
        schema = _DUCKDB.execute(f"DESCRIBE {quoted_table}").fetchall()
        row_count = _DUCKDB.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0]
        catalog.append(
            {
                "file_path": str(gold_path),
                "table_name": table_name,
                "row_count": int(row_count),
                "columns": [
                    {"name": row[0], "type": row[1]} for row in schema
                ],
            }
        )
    return json.dumps({"catalog": catalog}, default=str)


@tool
def execute_query_tool(sql: str) -> str:
    """Execute ANSI SQL against the loaded Gold catalog, allowing one retry after an error."""
    normalized_sql = sql.strip()
    attempt = _QUERY_ATTEMPTS.get(normalized_sql, 0) + 1
    _QUERY_ATTEMPTS[normalized_sql] = attempt
    try:
        result = _DUCKDB.execute(normalized_sql).fetchdf()
        rows = [
            {str(column): _json_value(value) for column, value in row.items()}
            for row in result.to_dict(orient="records")
        ]
        return json.dumps(
            {"sql": normalized_sql, "rows": rows, "columns": list(result.columns)},
            default=str,
        )
    except Exception as error:
        return json.dumps(
            {
                "sql": normalized_sql,
                "error": str(error),
                "attempt": attempt,
                "retry_allowed": attempt < 2,
            }
        )


def _json_from_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    if not isinstance(content, str) or not content.strip():
        return None
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _report_spec(messages: list[Any]) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    for message in reversed(messages):
        if type(message).__name__ == "AIMessage":
            parsed = _json_from_content(getattr(message, "content", ""))
            if parsed:
                spec = parsed
                break
    query_result = None
    for message in reversed(messages):
        if getattr(message, "name", None) != "execute_query_tool":
            continue
        query_result = _json_from_content(getattr(message, "content", ""))
        if query_result and "rows" in query_result:
            break
    if query_result and "rows" in query_result:
        spec.setdefault("sql", query_result.get("sql", ""))
        spec.setdefault("rows", query_result["rows"])
        spec.setdefault("columns", query_result.get("columns", []))
    if "rows" not in spec:
        raise ValueError("Reporter agent did not return query result rows")
    return spec


def _chart_html(rows: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    frame = pd.DataFrame(rows)
    if frame.empty or len(frame.columns) < 2:
        return "<p>No chartable query results were returned.</p>"
    columns = list(frame.columns)
    x_column = spec.get("x_column") or columns[0]
    y_column = spec.get("y_column")
    numeric_columns = list(frame.select_dtypes(include="number").columns)
    if y_column not in frame.columns:
        y_column = next((column for column in numeric_columns if column != x_column), None)
    if not y_column:
        return "<p>The query returned no numeric measure for charting.</p>"
    chart_type = str(spec.get("chart_type", "bar")).lower()
    if chart_type == "line":
        figure = px.line(frame, x=x_column, y=y_column, title=spec.get("chart_title"))
    elif chart_type == "pie":
        figure = px.pie(frame, names=x_column, values=y_column, title=spec.get("chart_title"))
    elif chart_type == "scatter":
        figure = px.scatter(frame, x=x_column, y=y_column, title=spec.get("chart_title"))
    else:
        figure = px.bar(frame, x=x_column, y=y_column, title=spec.get("chart_title"))
    return figure.to_html(full_html=False, include_plotlyjs=True)


def run_reporter(
    gold_paths: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    audit = AuditLogger(run_id)
    trace = AgentTrace("reporter", run_id)
    trace.set_input(
        gold_paths=gold_paths,
        business_intent=business_intent,
        task_description=task_description,
    )
    try:
        audit.log(
            "reporter",
            "started",
            gold_paths=gold_paths,
            business_intent=business_intent,
            task_description=task_description,
        )
        _QUERY_ATTEMPTS.clear()
        agent = create_react_agent(
            _make_llm(),
            [inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool],
            prompt=(
                "You are a reporting analyst. First inspect the Gold tables, then load "
                "them into DuckDB. Write one ANSI SQL query that answers the business "
                "question and execute it. If it errors, retry that SQL exactly once "
                "after using the returned error to correct it. Finish with ONLY valid "
                "JSON containing summary, sql, rows, chart_type, chart_title, "
                "x_column, and y_column. Choose bar, line, pie, or scatter based on "
                "the result. Keep the summary short and grounded in the query rows."
            ),
        )
        request = (
            f"Gold paths: {json.dumps(gold_paths)}\n"
            f"Business intent: {business_intent}\n"
            f"Task description: {task_description}\n"
            "Inspect, load, query, and prepare the final report JSON."
        )
        result = agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        spec = _report_spec(messages)
        rows = spec["rows"]
        report_json = {
            "run_id": run_id,
            "business_intent": business_intent,
            "task_description": task_description,
            "sql": spec.get("sql", ""),
            "summary": spec.get("summary", "The report contains the requested Gold analysis."),
            "rows": rows,
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"report_{run_id}.html"
        json_path = REPORTS_DIR / f"report_{run_id}.json"
        json_path.write_text(json.dumps(report_json, indent=2, default=str), encoding="utf-8")
        chart_html = _chart_html(rows, spec)
        table_html = pd.DataFrame(rows).to_html(index=False, escape=True)
        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Business Report {run_id}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.5rem;text-align:left}}th{{background:#f2f4f7}}pre{{white-space:pre-wrap;background:#f7f7f7;padding:1rem}}</style>
</head><body><h1>Executive Report</h1><h2>Summary</h2><p>{report_json["summary"]}</p>
<h2>Chart</h2>{chart_html}<h2>Underlying Data</h2>{table_html}<h2>SQL</h2><pre>{report_json["sql"]}</pre>
</body></html>"""
        report_path.write_text(html, encoding="utf-8")
        store_document(
            f"report_{run_id}",
            f"Business intent: {business_intent}\nSummary: {report_json['summary']}",
            {"run_id": run_id, "report_path": str(report_path)},
        )
        trace.set_output(report_path=str(report_path), json_path=str(json_path))
        audit.log("reporter", "completed", report_path=str(report_path), json_path=str(json_path))
        trace.complete()
        return str(report_path)
    except Exception as error:
        try:
            audit.log("reporter", "failed", error=str(error))
        except Exception:
            pass
        trace.fail(error)
        raise