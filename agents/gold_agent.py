import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import GOLD_DIR
from core.llm import _make_llm
from core.observability import AgentTrace


AGGREGATIONS = {"sum", "avg", "average", "count", "max", "min"}


def _read_rules(sttm_gold_path: str) -> pd.DataFrame:
    rules = pd.read_csv(sttm_gold_path, dtype=str).fillna("")
    required = {"source_col", "target_col", "transformation_logic"}
    missing = required.difference(rules.columns)
    if missing:
        raise ValueError(f"Gold STTM is missing required columns: {sorted(missing)}")
    return rules


def _table_name(path: str) -> str:
    return re.sub(r"_silver$", "", Path(path).stem)


def _key_columns(frames: list[pd.DataFrame]) -> list[str]:
    if not frames:
        return []
    shared = set(frames[0].columns)
    for frame in frames[1:]:
        shared.intersection_update(frame.columns)
    keys = [
        str(column)
        for column in shared
        if str(column).lower().endswith("_id")
        or str(column).lower().endswith("_key")
        or str(column).lower() in {"id", "key"}
    ]
    return sorted(keys)


def _explicit_join_keys(rules: pd.DataFrame, columns: pd.Index) -> list[str]:
    keys: list[str] = []
    for logic in rules["transformation_logic"]:
        matches = re.findall(
            r"(?:join\s+on|\bon)\s+([A-Za-z_][\w.]*)\s*(?:=|\band\b|\bwith\b)?\s*([A-Za-z_][\w.]*)?",
            logic,
            flags=re.IGNORECASE,
        )
        for left, right in matches:
            for candidate in (left, right):
                if candidate and candidate in columns and candidate not in keys:
                    keys.append(candidate)
    return keys


def _resolve_column(source_col: str, columns: pd.Index) -> str:
    if source_col in columns:
        return source_col
    unqualified = source_col.rsplit(".", 1)[-1]
    matches = [column for column in columns if str(column) == unqualified]
    if len(matches) == 1:
        return str(matches[0])
    qualified_matches = [
        column for column in columns if str(column).endswith(f"__{unqualified}")
    ]
    if len(qualified_matches) == 1:
        return str(qualified_matches[0])
    raise KeyError(f"Gold STTM source column not found: {source_col}")


def _merge_tables(
    silver_paths: list[str], rules: pd.DataFrame
) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in silver_paths]
    if not frames:
        raise ValueError("At least one Silver path is required")
    keys = _key_columns(frames)
    if len(frames) > 1 and not keys:
        keys = _explicit_join_keys(rules, frames[0].columns)
    if len(frames) > 1 and not keys:
        raise ValueError("Gold STTM did not identify a shared or explicit join key")

    merged = frames[0].copy()
    for index, frame in enumerate(frames[1:], start=1):
        overlapping = set(merged.columns).intersection(frame.columns) - set(keys)
        renamed = frame.rename(
            columns={
                column: f"{_table_name(silver_paths[index])}__{column}"
                for column in overlapping
            }
        )
        merged = merged.merge(renamed, how="outer", on=keys, suffixes=("", "_right"))
    return merged


def _aggregation(logic: str) -> str | None:
    match = re.search(r"\b(sum|avg|average|count|max|min)\b", logic.lower())
    if not match:
        return None
    function = match.group(1)
    return "mean" if function == "average" else function


def _materialize_table(frame: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    dimensions: list[tuple[str, str]] = []
    aggregations: dict[str, tuple[str, str]] = {}
    for rule in rules.to_dict("records"):
        source = _resolve_column(rule["source_col"].strip(), frame.columns)
        target = rule["target_col"].strip()
        if not target:
            raise ValueError(f"Gold STTM target column is empty for {source}")
        function = _aggregation(rule["transformation_logic"])
        if function:
            aggregations[target] = (source, function)
        else:
            dimensions.append((source, target))

    if not aggregations:
        result = frame[[source for source, _ in dimensions]].copy()
        result.columns = [target for _, target in dimensions]
        return result

    group_sources = []
    for source, _ in dimensions:
        if source not in group_sources:
            group_sources.append(source)
    if not group_sources:
        raise ValueError("Gold aggregations require at least one grouping column")

    grouped = frame.groupby(group_sources, dropna=False).agg(**aggregations).reset_index()
    rename_map = {
        source: target for source, target in dimensions if source in grouped.columns
    }
    return grouped.rename(columns=rename_map)


def _target_tables(rules: pd.DataFrame, sttm_gold_path: str) -> dict[str, pd.DataFrame]:
    if "target_table" not in rules.columns:
        return {re.sub(r"^sttm_gold_?|\.csv$", "", Path(sttm_gold_path).stem): rules}
    tables: dict[str, pd.DataFrame] = {}
    for target_table, table_rules in rules.groupby("target_table"):
        name = target_table.strip()
        if name:
            tables[name] = table_rules
    return tables or {"gold": rules}


@tool
def inspect_task_tool(silver_paths: list[str], sttm_gold_path: str) -> str:
    """Preview Silver schemas and list every Gold join and aggregation rule."""
    previews = []
    for silver_path in silver_paths:
        frame = pd.read_parquet(silver_path)
        previews.append(
            {
                "file_path": str(silver_path),
                "columns": [
                    {"name": str(column), "dtype": str(frame[column].dtype)}
                    for column in frame.columns
                ],
            }
        )
    rules = _read_rules(sttm_gold_path)
    return json.dumps(
        {"silver_tables": previews, "gold_rules": rules.to_dict(orient="records")},
        default=str,
    )


@tool
def gold_ingestion_tool(
    silver_paths: list[str], sttm_gold_path: str
) -> list[str]:
    """Merge Silver tables and materialize STTM-approved Gold aggregations."""
    rules = _read_rules(sttm_gold_path)
    merged = _merge_tables(silver_paths, rules)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for target_table, table_rules in _target_tables(rules, sttm_gold_path).items():
        result = _materialize_table(merged, table_rules)
        result.insert(0, "pk_gold_id", range(1, len(result) + 1))
        output_path = GOLD_DIR / f"{target_table}.parquet"
        result.to_parquet(output_path, engine="pyarrow", index=False)
        output_paths.append(str(output_path))
    return output_paths


def _paths_from_messages(messages: list[Any]) -> list[str]:
    for message in reversed(messages):
        if getattr(message, "name", None) != "gold_ingestion_tool":
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
    raise ValueError("Gold agent did not return Parquet output paths")


def run_gold(
    silver_paths: list[str],
    sttm_gold_path: str,
    business_intent: str,
    run_id: str,
) -> list[str]:
    audit = AuditLogger(run_id)
    trace = AgentTrace("gold", run_id)
    trace.set_input(
        silver_paths=silver_paths,
        sttm_gold_path=sttm_gold_path,
        business_intent=business_intent,
    )
    try:
        audit.log(
            "gold",
            "started",
            silver_paths=silver_paths,
            sttm_gold_path=sttm_gold_path,
            business_intent=business_intent,
        )
        agent = create_react_agent(
            _make_llm(),
            [inspect_task_tool, gold_ingestion_tool],
            prompt=(
                "You are the Gold analytics agent. First call inspect_task_tool to "
                "preview every Silver schema and read every Gold STTM join and "
                "aggregation rule. Then call gold_ingestion_tool exactly once. Apply "
                "only approved joins, renames, dimensions, and aggregations, and "
                "return the tool's output paths."
            ),
        )
        request = (
            f"Silver paths: {json.dumps(silver_paths)}\n"
            f"Gold STTM path: {sttm_gold_path}\n"
            f"Business intent: {business_intent}\n"
            "Inspect and materialize the analytics-ready Gold tables."
        )
        result = agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        output_paths = _paths_from_messages(messages)
        trace.set_output(gold_output_paths=output_paths)
        audit.log("gold", "completed", gold_output_paths=output_paths)
        trace.complete()
        return output_paths
    except Exception as error:
        try:
            audit.log("gold", "failed", error=str(error))
        except Exception:
            pass
        trace.fail(error)
        raise