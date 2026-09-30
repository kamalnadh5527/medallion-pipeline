import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import BRONZE_DIR
from core.llm import _make_llm
from core.observability import AgentTrace


def _read_rules(sttm_bronze_path: str) -> pd.DataFrame:
    rules = pd.read_csv(sttm_bronze_path, dtype=str).fillna("")
    required = {"source_col", "target_col", "transformation_logic"}
    missing = required.difference(rules.columns)
    if missing:
        raise ValueError(
            f"Bronze STTM is missing required columns: {sorted(missing)}"
        )
    return rules


def _source_column(source_col: str, columns: pd.Index) -> str:
    if source_col in columns:
        return source_col
    unqualified = source_col.rsplit(".", 1)[-1]
    matches = [column for column in columns if str(column) == unqualified]
    if len(matches) == 1:
        return str(matches[0])
    raise KeyError(f"STTM source column not found in CSV: {source_col}")


def _cast_column(frame: pd.DataFrame, column: str, logic: str) -> None:
    expression = logic.strip().lower()
    cast_match = re.search(
        r"(?:cast|convert|type)\s+(?:to|as)\s+([a-z][a-z0-9_ ]*)", expression
    )
    type_name = (cast_match.group(1) if cast_match else expression).strip()

    if any(token in type_name for token in ("datetime", "timestamp", "date")):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    elif any(token in type_name for token in ("bool", "boolean")):
        frame[column] = frame[column].astype("boolean")
    elif any(token in type_name for token in ("float", "double", "decimal", "number")):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Float64")
    elif any(token in type_name for token in ("int", "integer", "long")):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    elif any(token in type_name for token in ("str", "string", "text", "varchar", "char")):
        frame[column] = frame[column].astype("string")


def _apply_rules(frame: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    matched_rules: list[tuple[str, str, str]] = []
    for rule in rules.to_dict("records"):
        source = _source_column(rule["source_col"].strip(), frame.columns)
        target = rule["target_col"].strip()
        if not target:
            raise ValueError(f"Bronze STTM target column is empty for {source}")
        rename_map[source] = target
        matched_rules.append((source, target, rule["transformation_logic"]))

    frame = frame.rename(columns=rename_map)
    for source, target, logic in matched_rules:
        _cast_column(frame, target, logic)
    return frame


@tool
def inspect_task_tool(csv_paths: list[str], sttm_bronze_path: str) -> str:
    """Preview each raw CSV and list every Bronze STTM transformation rule."""
    previews = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        previews.append(
            {
                "file_path": str(csv_path),
                "shape": {"rows": len(frame), "columns": len(frame.columns)},
                "columns": [str(column) for column in frame.columns],
                "preview": frame.head(3).to_dict(orient="records"),
            }
        )

    rules = _read_rules(sttm_bronze_path)
    return json.dumps(
        {"files": previews, "bronze_rules": rules.to_dict(orient="records")},
        default=str,
    )


@tool
def bronze_ingestion_tool(
    csv_paths: list[str], sttm_bronze_path: str
) -> list[str]:
    """Apply Bronze renames and type casts, then write one Parquet per CSV."""
    rules = _read_rules(sttm_bronze_path)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for csv_path in csv_paths:
        frame = _apply_rules(pd.read_csv(csv_path), rules)
        input_path = Path(csv_path)
        output_path = BRONZE_DIR / f"{input_path.stem}_bronze.parquet"
        frame.to_parquet(output_path, engine="pyarrow", index=False)
        output_paths.append(str(output_path))
    return output_paths


def _paths_from_messages(messages: list[Any]) -> list[str]:
    for message in reversed(messages):
        if getattr(message, "name", None) != "bronze_ingestion_tool":
            continue
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        try:
            paths = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            continue
        if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
            return paths
    raise ValueError("Bronze agent did not return Parquet output paths")


def run_bronze(
    csv_paths: list[str],
    sttm_bronze_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    audit = AuditLogger(run_id)
    trace = AgentTrace("bronze", run_id)
    trace.set_input(
        csv_paths=csv_paths,
        sttm_bronze_path=sttm_bronze_path,
        task_description=task_description,
    )

    try:
        audit.log(
            "bronze",
            "started",
            csv_paths=csv_paths,
            sttm_bronze_path=sttm_bronze_path,
            task_description=task_description,
        )
        agent = create_react_agent(
            _make_llm(),
            [inspect_task_tool, bronze_ingestion_tool],
            prompt=(
                "You are the Bronze ingestion agent. First call inspect_task_tool to "
                "preview every CSV and read every Bronze STTM rule. Form an ingestion "
                "plan from that inspection, then call bronze_ingestion_tool exactly "
                "once. Bronze is raw-but-typed: apply only the approved STTM renames "
                "and casts, with no cleansing or business transformations."
            ),
        )
        request = (
            f"CSV paths: {json.dumps(csv_paths)}\n"
            f"Bronze STTM path: {sttm_bronze_path}\n"
            f"Task description: {task_description}\n"
            "Inspect the task and ingest these CSVs into Bronze Parquet files."
        )
        result = agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        output_paths = _paths_from_messages(messages)
        trace.set_output(bronze_output_paths=output_paths)
        audit.log("bronze", "completed", bronze_output_paths=output_paths)
        trace.complete()
        return output_paths
    except Exception as error:
        try:
            audit.log("bronze", "failed", error=str(error))
        except Exception:
            pass
        trace.fail(error)
        raise