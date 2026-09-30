import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import SILVER_DIR
from core.llm import _make_llm
from core.observability import AgentTrace


def _read_rules(sttm_silver_path: str) -> pd.DataFrame:
    rules = pd.read_csv(sttm_silver_path, dtype=str).fillna("")
    required = {"source_col", "target_col", "transformation_logic"}
    missing = required.difference(rules.columns)
    if missing:
        raise ValueError(f"Silver STTM is missing required columns: {sorted(missing)}")
    return rules


def _source_column(source_col: str, columns: pd.Index) -> str:
    if source_col in columns:
        return source_col
    unqualified = source_col.rsplit(".", 1)[-1]
    matches = [column for column in columns if str(column) == unqualified]
    if len(matches) == 1:
        return str(matches[0])
    raise KeyError(f"STTM source column not found in Bronze table: {source_col}")


def _cast_column(frame: pd.DataFrame, column: str, logic: str) -> None:
    expression = logic.lower()
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


def _fill_nulls(frame: pd.DataFrame, column: str, logic: str) -> None:
    expression = logic.lower()
    if not re.search(r"fill\s*na|fill\s*null|imput", expression):
        return
    if "mean" in expression and pd.api.types.is_numeric_dtype(frame[column]):
        frame[column] = frame[column].fillna(frame[column].mean())
    elif "median" in expression and pd.api.types.is_numeric_dtype(frame[column]):
        frame[column] = frame[column].fillna(frame[column].median())
    elif "mode" in expression:
        modes = frame[column].mode(dropna=True)
        if not modes.empty:
            frame[column] = frame[column].fillna(modes.iloc[0])
    else:
        value_match = re.search(r"(?:fill|replace)\s*(?:na|null)?\s*(?:with|as)\s+([^,;]+)", expression)
        if value_match:
            frame[column] = frame[column].fillna(value_match.group(1).strip())


def _clean_column(frame: pd.DataFrame, column: str, logic: str) -> None:
    expression = logic.lower()
    _fill_nulls(frame, column, expression)
    if re.search(r"drop\s*(?:na|null)|remove\s+(?:rows|records)\s+with\s+null", expression):
        frame.dropna(subset=[column], inplace=True)
    if re.search(r"text\s+normal|normalize\s+text|trim|lowercase|lower\s+case", expression):
        values = frame[column].astype("string").str.strip()
        if re.search(r"lowercase|lower\s+case|case[- ]?fold", expression):
            values = values.str.lower()
        frame[column] = values
    if re.search(r"date\s+standard|standardi[sz]e\s+date|datetime", expression):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    _cast_column(frame, column, expression)


def _apply_rules(frame: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    matched_rules: list[tuple[str, str, str]] = []
    for rule in rules.to_dict("records"):
        source = _source_column(rule["source_col"].strip(), frame.columns)
        target = rule["target_col"].strip()
        if not target:
            raise ValueError(f"Silver STTM target column is empty for {source}")
        rename_map[source] = target
        matched_rules.append((source, target, rule["transformation_logic"]))

    frame = frame.rename(columns=rename_map)
    frame = frame[[target for _, target, _ in matched_rules]]
    for _, target, logic in matched_rules:
        _clean_column(frame, target, logic)
    if any(
        re.search(r"drop[_ ]duplicates?", logic.lower())
        for _, _, logic in matched_rules
    ):
        frame = frame.drop_duplicates()
    return frame


def _table_name(bronze_path: str) -> str:
    name = Path(bronze_path).stem
    return re.sub(r"_bronze$", "", name)


@tool
def inspect_task_tool(bronze_paths: list[str], sttm_silver_path: str) -> str:
    """Preview Bronze schemas and list every Silver cleansing rule."""
    previews = []
    for bronze_path in bronze_paths:
        frame = pd.read_parquet(bronze_path)
        previews.append(
            {
                "file_path": str(bronze_path),
                "columns": [
                    {
                        "name": str(column),
                        "dtype": str(frame[column].dtype),
                        "null_count": int(frame[column].isna().sum()),
                        "sample_values": frame[column].dropna().head(3).tolist(),
                    }
                    for column in frame.columns
                ],
            }
        )
    rules = _read_rules(sttm_silver_path)
    return json.dumps(
        {"bronze_tables": previews, "silver_rules": rules.to_dict(orient="records")},
        default=str,
    )


@tool
def silver_ingestion_tool(
    bronze_paths: list[str], sttm_silver_path: str
) -> list[str]:
    """Cleanse Bronze tables according to Silver STTM rules and write Parquet files."""
    rules = _read_rules(sttm_silver_path)
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for bronze_path in bronze_paths:
        name = _table_name(bronze_path)
        frame = _apply_rules(pd.read_parquet(bronze_path), rules)
        frame.insert(0, f"pk_{name}_silver_id", range(1, len(frame) + 1))
        output_path = SILVER_DIR / f"{name}_silver.parquet"
        frame.to_parquet(output_path, engine="pyarrow", index=False)
        output_paths.append(str(output_path))
    return output_paths


def _paths_from_messages(messages: list[Any]) -> list[str]:
    for message in reversed(messages):
        if getattr(message, "name", None) != "silver_ingestion_tool":
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
    raise ValueError("Silver agent did not return Parquet output paths")


def run_silver(
    bronze_paths: list[str],
    sttm_silver_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    audit = AuditLogger(run_id)
    trace = AgentTrace("silver", run_id)
    trace.set_input(
        bronze_paths=bronze_paths,
        sttm_silver_path=sttm_silver_path,
        task_description=task_description,
    )
    try:
        audit.log(
            "silver",
            "started",
            bronze_paths=bronze_paths,
            sttm_silver_path=sttm_silver_path,
            task_description=task_description,
        )
        agent = create_react_agent(
            _make_llm(),
            [inspect_task_tool, silver_ingestion_tool],
            prompt=(
                "You are the Silver cleansing agent. First call inspect_task_tool to "
                "preview every Bronze Parquet schema and read every Silver STTM rule. "
                "Then call silver_ingestion_tool exactly once. Apply only approved "
                "cleansing rules, retain only STTM columns, and return the tool's "
                "output paths."
            ),
        )
        request = (
            f"Bronze paths: {json.dumps(bronze_paths)}\n"
            f"Silver STTM path: {sttm_silver_path}\n"
            f"Task description: {task_description}\n"
            "Inspect and cleanse these Bronze tables into Silver Parquet files."
        )
        result = agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        output_paths = _paths_from_messages(messages)
        trace.set_output(silver_output_paths=output_paths)
        audit.log("silver", "completed", silver_output_paths=output_paths)
        trace.complete()
        return output_paths
    except Exception as error:
        try:
            audit.log("silver", "failed", error=str(error))
        except Exception:
            pass
        trace.fail(error)
        raise