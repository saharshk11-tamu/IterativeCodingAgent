"""Shared hosted LLM provider interfaces."""

from llm.client import (
    LLMConfig,
    LLMProviderError,
    LLMResult,
    ModelSummary,
    OLLAMA_BASE_URL,
    ProviderKind,
    generate_text,
    list_models,
)

__all__ = [
    "LLMConfig",
    "LLMProviderError",
    "LLMResult",
    "ModelSummary",
    "OLLAMA_BASE_URL",
    "ProviderKind",
    "generate_text",
    "list_models",
]
