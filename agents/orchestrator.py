import json
import uuid
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import create_react_agent

from agents.bronze_agent import run_bronze
from agents.gold_agent import run_gold
from agents.profiler import run_profiler
from agents.reporter import run_reporter
from agents.silver_agent import run_silver
from agents.sttm_generator import run_sttm
from core.audit import AuditLogger
from core.config import (
    BRONZE_DIR,
    GOLD_DIR,
    PROFILES_DIR,
    REPORTS_DIR,
    SILVER_DIR,
    STTM_DIR,
)
from core.llm import _make_llm
from core.memory import store_document
from core.observability import AgentTrace
from core.state import PipelineState


def _verify_path(path: str, label: str) -> str:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} was not created: {path}")
    return path


def _copy_scratchpad(state: PipelineState, scratchpad: dict[str, Any]) -> None:
    state.scratchpad = dict(scratchpad)


def _phase_tool(
    name: str,
    goal: str,
    callback: Callable[[], dict[str, Any]],
    scratchpad: dict[str, Any],
) -> BaseTool:
    @tool(name)
    def phase_call(request: str = "") -> str:
        """Execute one approved pipeline operation for the supplied goal."""
        scratchpad.setdefault("goals", []).append(goal)
        try:
            result = callback()
            scratchpad.update(result)
            return json.dumps({"goal": goal, "result": result}, default=str)
        except Exception as error:
            scratchpad.setdefault("errors", []).append(
                {"goal": goal, "error": str(error)}
            )
            return json.dumps({"goal": goal, "error": str(error)})

    phase_call.__name__ = name
    return phase_call


def _run_supervisor(
    state: PipelineState,
    phase: str,
    tools: list[BaseTool],
    scratchpad: dict[str, Any],
    instruction: str,
) -> None:
    audit = AuditLogger(state.run_id)
    trace = AgentTrace("supervisor", state.run_id)
    trace.set_input(phase=phase, task_description=instruction)
    try:
        audit.log("supervisor", "phase_started", phase=phase, tools=[tool.name for tool in tools])
        agent = create_react_agent(
            _make_llm(),
            tools,
            prompt=(
                "You are the pipeline Supervisor. Execute the requested phase by "
                "calling each supplied tool in the stated order. Every call has a "
                "goal: use the goal described by the tool. If a tool returns an "
                "error, adapt the next call or retry that operation once. Verify "
                "that each required output exists before finishing, and summarize "
                "the completed outputs.\n" + instruction
            ),
        )
        result = agent.invoke({"messages": [("user", instruction)]})
        messages = result.get("messages", [])
        trace.extract_from_messages(messages)
        if scratchpad.get("errors"):
            raise RuntimeError(str(scratchpad["errors"][-1]))
        trace.set_output(**scratchpad)
        trace.complete()
        audit.log("supervisor", "phase_completed", phase=phase, outputs=scratchpad)
    except Exception as error:
        trace.fail(error)
        audit.log("supervisor", "phase_failed", phase=phase, error=str(error))
        raise


def _fail_state(state: PipelineState, error: Exception) -> PipelineState:
    state.error = str(error)
    state.status = "failed"
    AuditLogger(state.run_id).log("orchestrator", "failed", error=str(error))
    return state


def _phase_one(state: PipelineState) -> PipelineState:
    scratchpad: dict[str, Any] = {"phase": "phase_1"}
    try:
        tools = [
            _phase_tool(
                "run_profiler",
                "Profile every uploaded CSV and identify semantic fields and join keys.",
                lambda: {
                    "profile_path": _verify_path(
                        run_profiler(
                            state.uploaded_files,
                            state.business_intent,
                            state.run_id,
                            "Profile uploaded CSVs for the pipeline.",
                        ),
                        "Profile",
                    )
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_1_profile",
            tools,
            scratchpad,
            "Run profiling for the uploaded files and produce a profile JSON.",
        )
        state.profile_path = scratchpad["profile_path"]

        scratchpad = {"phase": "phase_1_sttm", "profile_path": state.profile_path}
        tools = [
            _phase_tool(
                "run_bronze_sttm",
                "Generate Bronze mappings from the approved profile context.",
                lambda: {
                    "sttm_bronze_path": _verify_path(
                        run_sttm(
                            "bronze",
                            state.profile_path,
                            state.business_intent,
                            state.run_id,
                        ),
                        "Bronze STTM",
                    )
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_1_bronze_sttm",
            tools,
            scratchpad,
            "Generate the Bronze STTM from the profile and pause for human approval.",
        )
        state.sttm_bronze_path = scratchpad["sttm_bronze_path"]
        _copy_scratchpad(state, scratchpad)
        state.hitl_approved = False
        state.status = "awaiting_bronze_approval"
        return state
    except Exception as error:
        _copy_scratchpad(state, scratchpad)
        return _fail_state(state, error)


def start_run(uploaded_files: list[str], business_intent: str) -> PipelineState:
    """Create a run, persist its intent, and execute Phase 1 through Bronze STTM."""
    run_id = str(uuid.uuid4())
    state = PipelineState(
        run_id=run_id,
        status="running",
        uploaded_files=uploaded_files,
        business_intent=business_intent,
        sttm_bronze_path="",
        sttm_silver_path="",
        sttm_gold_path="",
        bronze_output_paths=[],
        silver_output_paths=[],
        gold_output_paths=[],
    )
    AuditLogger(run_id).log(
        "orchestrator",
        "started",
        uploaded_files=uploaded_files,
        business_intent=business_intent,
    )
    store_document(
        f"business_intent_{run_id}",
        business_intent,
        {"run_id": run_id, "type": "business_intent"},
    )
    return _phase_one(state)


def _phase_two(state: PipelineState) -> PipelineState:
    scratchpad: dict[str, Any] = {"phase": "phase_2"}
    try:
        tools = [
            _phase_tool(
                "run_bronze",
                "Ingest the approved Bronze STTM into typed Bronze Parquet files.",
                lambda: {
                    "bronze_output_paths": [
                        _verify_path(path, "Bronze output")
                        for path in run_bronze(
                            state.uploaded_files,
                            state.sttm_bronze_path,
                            state.run_id,
                            "Execute the approved Bronze STTM.",
                        )
                    ]
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_2_bronze",
            tools,
            scratchpad,
            "Execute the approved Bronze STTM and verify all Parquet outputs.",
        )
        state.bronze_output_paths = scratchpad["bronze_output_paths"]
        scratchpad = {"phase": "phase_2_sttm", "bronze_output_paths": state.bronze_output_paths}
        tools = [
            _phase_tool(
                "run_silver_sttm",
                "Generate Silver cleansing mappings from Bronze Parquet schemas.",
                lambda: {
                    "sttm_silver_path": _verify_path(
                        run_sttm(
                            "silver",
                            state.bronze_output_paths,
                            state.business_intent,
                            state.run_id,
                        ),
                        "Silver STTM",
                    )
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_2_silver_sttm",
            tools,
            scratchpad,
            "Generate the Silver STTM and pause for human approval.",
        )
        state.sttm_silver_path = scratchpad["sttm_silver_path"]
        _copy_scratchpad(state, scratchpad)
        state.hitl_approved = False
        state.status = "awaiting_silver_approval"
        return state
    except Exception as error:
        _copy_scratchpad(state, scratchpad)
        return _fail_state(state, error)


def _phase_three(state: PipelineState) -> PipelineState:
    scratchpad: dict[str, Any] = {"phase": "phase_3"}
    try:
        tools = [
            _phase_tool(
                "run_silver",
                "Cleanse Bronze data with the approved Silver STTM.",
                lambda: {
                    "silver_output_paths": [
                        _verify_path(path, "Silver output")
                        for path in run_silver(
                            state.bronze_output_paths,
                            state.sttm_silver_path,
                            state.run_id,
                            "Execute the approved Silver STTM.",
                        )
                    ]
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_3_silver",
            tools,
            scratchpad,
            "Execute the approved Silver STTM and verify all Parquet outputs.",
        )
        state.silver_output_paths = scratchpad["silver_output_paths"]
        scratchpad = {"phase": "phase_3_sttm", "silver_output_paths": state.silver_output_paths}
        tools = [
            _phase_tool(
                "run_gold_sttm",
                "Generate Gold join and aggregation mappings from Silver schemas.",
                lambda: {
                    "sttm_gold_path": _verify_path(
                        run_sttm(
                            "gold",
                            state.silver_output_paths,
                            state.business_intent,
                            state.run_id,
                        ),
                        "Gold STTM",
                    )
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_3_gold_sttm",
            tools,
            scratchpad,
            "Generate the Gold STTM and pause for human approval.",
        )
        state.sttm_gold_path = scratchpad["sttm_gold_path"]
        _copy_scratchpad(state, scratchpad)
        state.hitl_approved = False
        state.status = "awaiting_gold_approval"
        return state
    except Exception as error:
        _copy_scratchpad(state, scratchpad)
        return _fail_state(state, error)


def _phase_four(state: PipelineState) -> PipelineState:
    scratchpad: dict[str, Any] = {"phase": "phase_4"}
    try:
        tools = [
            _phase_tool(
                "run_gold",
                "Materialize analytics-ready Gold tables from the approved Gold STTM.",
                lambda: {
                    "gold_output_paths": [
                        _verify_path(path, "Gold output")
                        for path in run_gold(
                            state.silver_output_paths,
                            state.sttm_gold_path,
                            state.business_intent,
                            state.run_id,
                        )
                    ]
                },
                scratchpad,
            ),
            _phase_tool(
                "run_reporter",
                "Query Gold tables and create the human-readable business report.",
                lambda: {
                    "report_path": _verify_path(
                        run_reporter(
                            scratchpad["gold_output_paths"],
                            state.business_intent,
                            state.run_id,
                            "Answer the business question from Gold data.",
                        ),
                        "Report",
                    )
                },
                scratchpad,
            ),
        ]
        _run_supervisor(
            state,
            "phase_4_gold_report",
            tools,
            scratchpad,
            "Materialize Gold, then create and verify the final report.",
        )
        state.gold_output_paths = scratchpad["gold_output_paths"]
        state.report_path = scratchpad["report_path"]
        _copy_scratchpad(state, scratchpad)
        state.hitl_approved = True
        state.status = "completed"
        return state
    except Exception as error:
        _copy_scratchpad(state, scratchpad)
        return _fail_state(state, error)


def approve_and_continue(state: PipelineState, layer: str) -> PipelineState:
    """Record HITL approval for a layer and execute the next workflow phase."""
    expected = {
        "bronze": ("awaiting_bronze_approval", _phase_two),
        "silver": ("awaiting_silver_approval", _phase_three),
        "gold": ("awaiting_gold_approval", _phase_four),
    }
    normalized_layer = layer.strip().lower()
    if normalized_layer not in expected:
        raise ValueError(f"Unsupported approval layer: {layer!r}")
    required_status, next_phase = expected[normalized_layer]
    if state.status != required_status:
        raise ValueError(
            f"Cannot approve {normalized_layer} while pipeline status is {state.status!r}"
        )
    state.hitl_approved = True
    AuditLogger(state.run_id).log(
        "orchestrator",
        "hitl_approved",
        layer=normalized_layer,
    )
    return next_phase(state)