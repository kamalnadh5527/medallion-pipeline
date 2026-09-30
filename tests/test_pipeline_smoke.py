import json
from pathlib import Path

import pandas as pd

import agents.orchestrator as orchestrator
from tools.generate_sample_data import generate_sample_data


def test_cli_style_pipeline_creates_expected_artifacts(tmp_path, monkeypatch) -> None:
    input_paths = generate_sample_data()
    output_paths = {
        "profile": tmp_path / "profile.json",
        "bronze_sttm": tmp_path / "sttm_bronze.csv",
        "silver_sttm": tmp_path / "sttm_silver.csv",
        "gold_sttm": tmp_path / "sttm_gold.csv",
        "bronze": tmp_path / "sales_bronze.parquet",
        "silver": tmp_path / "sales_silver.parquet",
        "gold": tmp_path / "sales.parquet",
        "report": tmp_path / "report.html",
    }
    mapping_csv = "source_col,target_col,transformation_logic\nvalue,value,sum\n"

    def fake_profile(*args):
        output_paths["profile"].write_text(json.dumps({"files": []}), encoding="utf-8")
        return str(output_paths["profile"])

    def fake_sttm(layer, *args):
        path = output_paths[f"{layer}_sttm"]
        path.write_text(mapping_csv, encoding="utf-8")
        return str(path)

    def fake_bronze(*args):
        pd.DataFrame({"value": [1]}).to_parquet(output_paths["bronze"], index=False)
        return [str(output_paths["bronze"])]

    def fake_silver(*args):
        pd.DataFrame({"value": [1]}).to_parquet(output_paths["silver"], index=False)
        return [str(output_paths["silver"])]

    def fake_gold(*args):
        pd.DataFrame({"total": [1]}).to_parquet(output_paths["gold"], index=False)
        return [str(output_paths["gold"])]

    def fake_reporter(*args):
        output_paths["report"].write_text("<html>report</html>", encoding="utf-8")
        return str(output_paths["report"])

    def fake_supervisor(state, phase, tools, scratchpad, instruction):
        for phase_tool in tools:
            phase_tool.invoke({"request": instruction})

    monkeypatch.setattr(orchestrator, "_make_llm", lambda: object())
    monkeypatch.setattr(orchestrator, "store_document", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "run_profiler", fake_profile)
    monkeypatch.setattr(orchestrator, "run_sttm", fake_sttm)
    monkeypatch.setattr(orchestrator, "run_bronze", fake_bronze)
    monkeypatch.setattr(orchestrator, "run_silver", fake_silver)
    monkeypatch.setattr(orchestrator, "run_gold", fake_gold)
    monkeypatch.setattr(orchestrator, "run_reporter", fake_reporter)
    monkeypatch.setattr(orchestrator, "_run_supervisor", fake_supervisor)

    state = orchestrator.start_run([str(path) for path in input_paths], "What are total sales?")
    state = orchestrator.approve_and_continue(state, "bronze")
    state = orchestrator.approve_and_continue(state, "silver")
    state = orchestrator.approve_and_continue(state, "gold")

    assert state.status == "completed"
    assert all(path.exists() for path in output_paths.values())