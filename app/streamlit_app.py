from pathlib import Path

import pandas as pd
import streamlit as st

from agents.orchestrator import approve_and_continue, start_run
from core.config import LANDING_DIR


st.set_page_config(page_title="Medallion Pipeline", page_icon="📊", layout="wide")


STATUS_LABELS = {
    "initialized": "Ready to start",
    "running": "Running",
    "awaiting_bronze_approval": "Waiting for Bronze STTM approval",
    "awaiting_silver_approval": "Waiting for Silver STTM approval",
    "awaiting_gold_approval": "Waiting for Gold STTM approval",
    "completed": "Pipeline complete",
    "failed": "Pipeline failed",
}


def _save_uploads(uploaded_files: list) -> list[str]:
    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for uploaded_file in uploaded_files:
        safe_name = Path(uploaded_file.name).name
        destination = LANDING_DIR / safe_name
        destination.write_bytes(uploaded_file.getvalue())
        paths.append(str(destination))
    return paths


def _sttm_path_for_state(state) -> tuple[str | None, str | None]:
    gates = {
        "awaiting_bronze_approval": ("Bronze", state.sttm_bronze_path),
        "awaiting_silver_approval": ("Silver", state.sttm_silver_path),
        "awaiting_gold_approval": ("Gold", state.sttm_gold_path),
    }
    return gates.get(state.status, (None, None))


def _render_sttm_gate(state) -> None:
    layer, sttm_path = _sttm_path_for_state(state)
    if not layer or not sttm_path:
        return

    st.subheader(f"{layer} STTM review")
    st.caption("Review or edit the mapping below before approving this phase.")
    try:
        mappings = pd.read_csv(sttm_path, dtype=str).fillna("")
    except Exception as error:
        st.error(f"Unable to read the {layer} STTM: {error}")
        return

    edited_mappings = st.data_editor(
        mappings,
        key=f"sttm_editor_{layer.lower()}_{state.run_id}",
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
    )
    if st.button(f"Approve {layer} STTM and continue", type="primary"):
        try:
            edited_mappings.to_csv(sttm_path, index=False)
            st.session_state.pipeline_state = approve_and_continue(state, layer.lower())
            st.rerun()
        except Exception as error:
            st.error(f"Unable to continue after {layer} approval: {error}")


def _render_progress(state) -> None:
    st.subheader("Pipeline progress")
    phases = [
        ("Profile and Bronze STTM", "awaiting_bronze_approval"),
        ("Bronze and Silver STTM", "awaiting_silver_approval"),
        ("Silver and Gold STTM", "awaiting_gold_approval"),
        ("Gold and report", "completed"),
    ]
    current_phase = {
        "awaiting_bronze_approval": 0,
        "awaiting_silver_approval": 1,
        "awaiting_gold_approval": 2,
    }.get(state.status)
    for index, (label, completion_status) in enumerate(phases):
        if state.status == "completed" or (
            current_phase is not None and index < current_phase
        ):
            marker = "Complete"
        elif current_phase == index or state.status == completion_status:
            marker = "Current"
        else:
            marker = "Pending"
        st.write(f"**{label}:** {marker}")
    st.info(STATUS_LABELS.get(state.status, state.status))


def _render_outputs(state) -> None:
    if state.profile_path:
        st.caption(f"Profile: {state.profile_path}")
    if state.bronze_output_paths:
        st.caption(f"Bronze outputs: {len(state.bronze_output_paths)}")
    if state.silver_output_paths:
        st.caption(f"Silver outputs: {len(state.silver_output_paths)}")
    if state.gold_output_paths:
        st.caption(f"Gold outputs: {len(state.gold_output_paths)}")


def main() -> None:
    st.title("Medallion Data Pipeline")
    st.write("Upload CSV data, approve each generated mapping, and review the final analysis.")
    st.session_state.setdefault("pipeline_state", None)

    state = st.session_state.pipeline_state
    if state is None:
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )
        business_intent = st.text_area(
            "Business question",
            placeholder="What should the final report explain?",
        )
        if st.button("Start pipeline", type="primary"):
            if not uploaded_files:
                st.warning("Upload at least one CSV file before starting.")
            elif not business_intent.strip():
                st.warning("Enter a business question before starting.")
            else:
                try:
                    paths = _save_uploads(uploaded_files)
                    st.session_state.pipeline_state = start_run(
                        paths,
                        business_intent.strip(),
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Unable to start the pipeline: {error}")
        return

    _render_progress(state)
    _render_outputs(state)
    if state.error:
        st.error(state.error)
    if state.scratchpad:
        with st.expander("Latest supervisor notes"):
            st.json(state.scratchpad)

    if state.status.startswith("awaiting_"):
        _render_sttm_gate(state)
    elif state.status == "completed" and state.report_path:
        st.subheader("Final report")
        st.components.v1.html(
            open(state.report_path, encoding="utf-8").read(),
            height=900,
            scrolling=True,
        )


if __name__ == "__main__":
    main()
