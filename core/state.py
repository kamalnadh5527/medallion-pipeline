from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    # Run metadata
    run_id: str = Field(default="", description="Unique identifier for this pipeline run.")
    status: str = Field(default="initialized", description="Current pipeline status.")

    # Phase 1 - Ingestion
    uploaded_files: list[str] = Field(
        default_factory=list,
        description="Paths to uploaded CSV files.",
    )
    business_intent: str = Field(
        default="",
        description="User's business question in plain English.",
    )

    # Phase 2 - Profiling
    profile_path: str = Field(
        default="",
        description="Path to the data profile JSON file.",
    )

    # Phase 3 - STTM
    sttm_bronze_path: str = Field(
        default="",
        description="Path to Bronze layer transformation rules.",
    )
    sttm_silver_path: str = Field(
        default="",
        description="Path to Silver layer transformation rules.",
    )
    sttm_gold_path: str = Field(
        default="",
        description="Path to Gold layer transformation rules.",
    )
    hitl_approved: bool = Field(
        default=False,
        description="Whether a human has approved the transformation rules.",
    )

    # Phase 4 - Execution
    bronze_output_paths: list[str] = Field(
        default_factory=list,
        description="Paths to Bronze Parquet output files.",
    )
    silver_output_paths: list[str] = Field(
        default_factory=list,
        description="Paths to Silver Parquet output files.",
    )
    gold_output_paths: list[str] = Field(
        default_factory=list,
        description="Paths to Gold Parquet output files.",
    )

    # Phase 5 - Reporting
    report_path: str = Field(default="", description="Path to the final HTML report.")

    # Error handling
    error: Optional[str] = Field(
        default=None,
        description="Error message if the pipeline run fails.",
    )
    scratchpad: dict[str, Any] = Field(
        default_factory=dict,
        description="Supervisor notes and outputs copied after each phase.",
    )