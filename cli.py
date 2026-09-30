import argparse
import traceback

from agents.orchestrator import approve_and_continue, start_run
from core.memory import store_document


def _print_artifacts(state, phase: str) -> None:
    print(f"\n{phase}")
    if state.profile_path:
        print(f"profile: {state.profile_path}")
    if state.sttm_bronze_path:
        print(f"bronze STTM: {state.sttm_bronze_path}")
    if state.sttm_silver_path:
        print(f"silver STTM: {state.sttm_silver_path}")
    if state.sttm_gold_path:
        print(f"gold STTM: {state.sttm_gold_path}")
    for label, paths in (
        ("bronze parquet", state.bronze_output_paths),
        ("silver parquet", state.silver_output_paths),
        ("gold parquet", state.gold_output_paths),
    ):
        for path in paths:
            print(f"{label}: {path}")
    if state.report_path:
        print(f"report: {state.report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the medallion pipeline end-to-end without Streamlit."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more input CSV paths.",
    )
    parser.add_argument(
        "--intent",
        required=True,
        help="Business question for the final report.",
    )
    args = parser.parse_args()

    try:
        store_document(
            f"cli_business_intent_{args.intent[:32]}",
            args.intent,
            {"source": "cli", "business_intent": args.intent},
        )
        state = start_run(args.files, args.intent)
        _print_artifacts(state, "Phase 1: profile and Bronze STTM")
        if state.error:
            raise RuntimeError(state.error)

        state = approve_and_continue(state, "bronze")
        _print_artifacts(state, "Phase 2: Bronze and Silver STTM")
        if state.error:
            raise RuntimeError(state.error)

        state = approve_and_continue(state, "silver")
        _print_artifacts(state, "Phase 3: Silver and Gold STTM")
        if state.error:
            raise RuntimeError(state.error)

        state = approve_and_continue(state, "gold")
        _print_artifacts(state, "Phase 4: Gold and report")
        if state.error:
            raise RuntimeError(state.error)
        print(f"\nFinal report: {state.report_path}")
    except Exception as error:
        print(f"Pipeline failed: {error}")
        traceback.print_exc()
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
