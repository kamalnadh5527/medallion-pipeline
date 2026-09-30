from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GITHUB_BASE_URL,
    GITHUB_MODEL,
    GITHUB_TOKEN,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def _make_llm(temperature=0.0):
    if LLM_PROVIDER == "anthropic":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required for the anthropic provider")
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=temperature,
        )

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for the groq provider")
        options = {}
        if GROQ_MODEL.startswith("openai/gpt-oss"):
            options["reasoning_format"] = "hidden"
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
            **options,
        )

    if LLM_PROVIDER == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for the gemini provider")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=temperature,
        )

    if LLM_PROVIDER == "github":
        if not GITHUB_TOKEN:
            raise ValueError("GITHUB_TOKEN is required for the github provider")
        return ChatOpenAI(
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
            api_key=GITHUB_TOKEN,
            temperature=temperature,
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER!r}")