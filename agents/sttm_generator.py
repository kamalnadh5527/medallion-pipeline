import csv
import io
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import STTM_DIR
from core.llm import _make_llm
from core.observability import AgentTrace


STTM_COLUMNS = ("source_col", "target_col", "transformation_logic")
VALID_LAYERS = {"bronze", "silver", "gold"}


def _normalize_source_paths(source_path: str | list[str]) -> list[str]:
    if isinstance(source_path, str):
        try:
            parsed = json.loads(source_path)
        except json.JSONDecodeError:
            parsed = source_path
        source_paths = parsed if isinstance(parsed, list) else [parsed]
    else:
        source_paths = source_path

    paths = [str(path) for path in source_paths]
    if not paths:
        raise ValueError("At least one source path is required")
    return paths


def _parquet_files(source_path: str) -> list[Path]:
    path = Path(source_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    raise FileNotFoundError(f"Source path does not exist: {source_path}")


@tool
def inspect_context_tool(layer: str, source_path: str) -> str:
    """Preview a profile JSON for Bronze or Parquet schemas for Silver and Gold."""
    normalized_layer = layer.strip().lower()
    if normalized_layer not in VALID_LAYERS:
        raise ValueError(f"Unsupported STTM layer: {layer!r}")

    source_paths = _normalize_source_paths(source_path)
    if normalized_layer == "bronze":
        contexts = []
        for path in source_paths:
            with Path(path).open("r", encoding="utf-8") as file:
                contexts.append({"source_path": path, "profile": json.load(file)})
        return json.dumps({"layer": normalized_layer, "sources": contexts}, default=str)

    from pyarrow import parquet

    contexts = []
    for source in source_paths:
        files = _parquet_files(source)
        if not files:
            raise FileNotFoundError(f"No Parquet files found at: {source}")
        for parquet_path in files:
            schema = parquet.ParquetFile(parquet_path).schema_arrow
            contexts.append(
                {
                    "source_path": str(parquet_path),
                    "columns": [
                        {"name": field.name, "dtype": str(field.type)}
                        for field in schema
                    ],
                }
            )
    return json.dumps({"layer": normalized_layer, "sources": contexts})


def _mappings_to_csv(mappings: list[dict[str, str]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=STTM_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for index, mapping in enumerate(mappings):
        row = {column: str(mapping.get(column, "")).strip() for column in STTM_COLUMNS}
        if not all(row.values()):
            raise ValueError(
                f"Mapping {index} must provide source_col, target_col, and transformation_logic"
            )
        writer.writerow(row)
    if not mappings:
        raise ValueError("At least one STTM mapping is required")
    return output.getvalue()


@tool
def generate_bronze_sttm_tool(mappings: list[dict[str, str]]) -> str:
    """Validate Bronze ingestion mappings; use profile columns and explicit type casts."""
    return _mappings_to_csv(mappings)


@tool
def generate_silver_sttm_tool(mappings: list[dict[str, str]]) -> str:
    """Validate Silver mappings with explicit null, deduplication, casting, and normalization logic."""
    return _mappings_to_csv(mappings)


@tool
def generate_gold_sttm_tool(mappings: list[dict[str, str]]) -> str:
    """Validate Gold mappings with business-intent-driven joins and aggregate expressions."""
    return _mappings_to_csv(mappings)


def _csv_from_messages(messages: list[Any], layer: str) -> str:
    generator_name = f"generate_{layer}_sttm_tool"
    candidates = [
        getattr(message, "content", "")
        for message in reversed(messages)
        if getattr(message, "name", None) == generator_name
    ]
    candidates.extend(
        getattr(message, "content", "")
        for message in reversed(messages)
        if type(message).__name__ == "AIMessage"
    )

    for content in candidates:
        if not isinstance(content, str) or not content.strip():
            continue
        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            rows = list(csv.DictReader(io.StringIO(candidate)))
        except csv.Error:
            continue
        if not rows or set(rows[0]) != set(STTM_COLUMNS):
            continue
        return _mappings_to_csv(rows)
    raise ValueError("STTM agent did not return valid CSV mapping rows")


def run_sttm(
    layer: str,
    source_path_or_paths: str | list[str],
    business_intent: str,
    run_id: str,
) -> str:
    normalized_layer = layer.strip().lower()
    if normalized_layer not in VALID_LAYERS:
        raise ValueError(f"Unsupported STTM layer: {layer!r}")

    source_paths = _normalize_source_paths(source_path_or_paths)
    agent_name = f"sttm_{normalized_layer}"
    audit = AuditLogger(run_id)
    trace = AgentTrace(agent_name, run_id)
    trace.set_input(
        layer=normalized_layer,
        source_paths=source_paths,
        business_intent=business_intent,
    )

    try:
        audit.log(
            agent_name,
            "started",
            source_paths=source_paths,
            business_intent=business_intent,
        )
        system_prompt = (
            "You generate Source-to-Target Mapping rules from the inspected upstream "
            "context. First call inspect_context_tool for the requested layer and all "
            "source paths. Then call exactly the matching generate_<layer>_sttm_tool "
            "with mapping objects containing source_col, target_col, and "
            "transformation_logic. Return only the CSV produced by that tool with "
            "the exact header source_col,target_col,transformation_logic. Use "
            "qualified source table names when column names repeat across inputs. "
            "Bronze means direct raw-to-typed ingestion mappings with source types "
            "grounded in profile.json. Silver mappings must describe concrete null "
            "handling, deduplication, type casting, and text/date normalization "
            "where appropriate. Gold mappings must use the business intent to "
            "describe joins across available Silver tables and relevant sum, avg, "
            "count, max, or min aggregations; do not invent unavailable fields. "
            "For Gold, source_col may be a clearly written multi-column expression "
            "when a target depends on a join or aggregation."
        )
        generator_tool = {
            "bronze": generate_bronze_sttm_tool,
            "silver": generate_silver_sttm_tool,
            "gold": generate_gold_sttm_tool,
        }[normalized_layer]
        agent = create_react_agent(
            _make_llm(),
            [inspect_context_tool, generator_tool],
            prompt=system_prompt,
        )
        request = (
            f"Layer: {normalized_layer}\n"
            f"Source path(s): {json.dumps(source_paths)}\n"
            f"Business intent: {business_intent}\n"
            "Inspect context and produce the STTM CSV for this layer."
        )
        result = agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        sttm_csv = _csv_from_messages(messages, normalized_layer)

        STTM_DIR.mkdir(parents=True, exist_ok=True)
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id))
        sttm_path = Path(STTM_DIR) / f"sttm_{normalized_layer}_{safe_run_id}.csv"
        sttm_path.write_text(sttm_csv, encoding="utf-8", newline="")
        trace.set_output(sttm_path=str(sttm_path), layer=normalized_layer)
        audit.log(agent_name, "sttm_saved", sttm_path=str(sttm_path))
        trace.complete()
        audit.log(agent_name, "completed", sttm_path=str(sttm_path))
        return str(sttm_path)
    except Exception as error:
        try:
            audit.log(agent_name, "failed", error=str(error))
        except Exception:
            pass
        trace.fail(error)
        raise