from core.state import PipelineState


def test_pipeline_state_defaults() -> None:
    state = PipelineState()

    assert state.run_id == ""
    assert state.status == "initialized"
    assert state.uploaded_files == []
    assert state.business_intent == ""
    assert state.sttm_bronze_path == ""
    assert state.sttm_silver_path == ""
    assert state.sttm_gold_path == ""
    assert state.hitl_approved is False
    assert state.bronze_output_paths == []
    assert state.silver_output_paths == []
    assert state.gold_output_paths == []
    assert state.report_path == ""
    assert state.error is None
    assert state.scratchpad == {}