from core import config


def test_config_directories_exist() -> None:
    directories = (
        config.DATA_DIR,
        config.LANDING_DIR,
        config.PROFILES_DIR,
        config.STTM_DIR,
        config.BRONZE_DIR,
        config.SILVER_DIR,
        config.GOLD_DIR,
        config.REPORTS_DIR,
        config.AUDIT_DIR,
        config.CHROMA_DIR,
    )
    assert all(directory.is_dir() for directory in directories)


def test_selected_llm_provider_is_supported() -> None:
    assert config.LLM_PROVIDER in {"anthropic", "github", "openai", "groq", "gemini"}