import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.audit import AuditLogger
from core.config import PROFILES_DIR
from core.llm import _make_llm
from core.observability import AgentTrace


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


@tool
def inspect_files_tool(csv_paths: list[str]) -> str:
    """Preview CSV shapes, column names, inferred types, and three values per column."""
    previews = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        previews.append(
            {
                "file_path": str(csv_path),
                "shape": {"rows": len(frame), "columns": len(frame.columns)},
                "columns": [
                    {
                        "name": str(column),
                        "dtype": str(frame[column].dtype),
                        "sample_values": [
                            _json_value(value)
                            for value in frame[column].head(3).tolist()
                        ],
                    }
                    for column in frame.columns
                ],
            }
        )
    return json.dumps({"files": previews}, ensure_ascii=False, default=str)


@tool
def profiler_tool(csv_paths: list[str]) -> str:
    """Compute full null, cardinality, sample, and numeric statistics for CSV files."""
    profiles = []
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        columns = []
        for column in frame.columns:
            series = frame[column]
            stats = {
                "name": str(column),
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "null_percent": (
                    float(series.isna().mean() * 100) if len(series) else 0.0
                ),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [
                    _json_value(value) for value in series.dropna().head(5).tolist()
                ],
            }
            if (
                pd.api.types.is_numeric_dtype(series)
                and not pd.api.types.is_bool_dtype(series)
            ):
                stats.update(
                    {
                        "min": _json_value(series.min()),
                        "max": _json_value(series.max()),
                        "mean": _json_value(series.mean()),
                    }
                )
            columns.append(stats)

        profiles.append(
            {
                "file_path": str(csv_path),
                "shape": {"rows": len(frame), "columns": len(frame.columns)},
                "columns": columns,
            }
        )
    return json.dumps({"files": profiles}, ensure_ascii=False, default=str)


def _profile_from_messages(messages: list[Any]) -> dict[str, Any]:
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, list):
            content = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
            )
        if not isinstance(content, str) or not content.strip():
            continue

        candidate = content.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            profile = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end < start:
                continue
            try:
                profile = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
        if isinstance(profile, dict):
            return profile

    raise ValueError("Profiler agent did not return a valid JSON profile")


def run_profiler(
    csv_paths: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    audit = AuditLogger(run_id)
    trace = AgentTrace("profiler", run_id)
    trace.set_input(
        csv_paths=csv_paths,
        business_intent=business_intent,
        task_description=task_description,
    )

    try:
        audit.log(
            "profiler",
            "started",
            csv_paths=csv_paths,
            business_intent=business_intent,
            task_description=task_description,
        )
        system_prompt = (
            "You are a data profiling agent. For every request, call "
            "inspect_files_tool first and use its preview to plan; only then call "
            "profiler_tool for complete statistics. Use the statistics, column "
            "names, sample values, and the stated business intent to identify likely "
            "semantic meanings, cross-dataset join keys, and data-quality notes. "
            "Return only one valid JSON object with a top-level 'files' array and a "
            "top-level 'join_keys' array. Each file must include 'file_path' and a "
            "'columns' array. Each column must include 'name', 'semantic_meanings' "
            "(an array of concise meanings), 'join_keys' (an array of matching "
            "file/column references or an empty array), and 'quality_notes' "
            "(an array, empty when none apply), along with its computed statistics. "
            "For each top-level join key, identify the participating files and "
            "columns. Do not invent statistics or claim uncertain joins as facts."
        )
        agent = create_react_agent(
            _make_llm(),
            [inspect_files_tool, profiler_tool],
            prompt=system_prompt,
        )
        request = (
            f"CSV paths: {json.dumps(csv_paths)}\n"
            f"Business intent: {business_intent}\n"
            f"Task description: {task_description}\n"
            "Analyze these datasets and produce the required combined JSON profile."
        )
        result = agent.invoke({"messages": [("user", request)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        profile = _profile_from_messages(messages)

        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profile_path = Path(PROFILES_DIR) / (
            f"profile_combined_{datetime.now().strftime('%Y%m%d')}.json"
        )
        profile_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        trace.set_output(profile_path=str(profile_path))
        audit.log(
            "profiler",
            "profile_saved",
            profile_path=str(profile_path),
        )
        trace.complete()
        audit.log("profiler", "completed", profile_path=str(profile_path))
        return str(profile_path)
    except Exception as error:
        try:
            audit.log("profiler", "failed", error=str(error))
        except Exception:
            pass
        trace.fail(error)
        raise