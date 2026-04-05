"""Hosted LLM provider adapters used by the Textual app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

ProviderKind = Literal["openai", "anthropic", "gemini"]

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODELS_URL = f"{GEMINI_BASE_URL}/models"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_OUTPUT_TOKENS = 4096


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderKind
    model: str
    api_key: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: ProviderKind
    model: str
    error: str | None = None


@dataclass(frozen=True)
class ModelSummary:
    id: str
    display_name: str | None = None


class LLMProviderError(RuntimeError):
    """A normalized provider error surfaced to the UI."""


class ProviderAdapter(Protocol):
    def list_models(self, config: LLMConfig) -> list[ModelSummary]: ...
    def generate_text(self, config: LLMConfig, messages: list[dict]) -> LLMResult: ...


class _BaseAdapter:
    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def _request_json(
        self,
        provider: ProviderKind,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict | None = None,
        timeout: float = 30.0,
    ) -> dict:
        try:
            with httpx.Client(transport=self._transport, timeout=timeout) as client:
                response = client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                )
        except httpx.TimeoutException as exc:
            raise LLMProviderError(f"{provider} request timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"{provider} request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = _extract_error_detail(response)
            if response.status_code in {401, 403}:
                prefix = "authentication failed"
            elif response.status_code == 429:
                prefix = "rate limit exceeded"
            elif response.status_code >= 500:
                prefix = "server error"
            else:
                prefix = "request failed"
            raise LLMProviderError(
                f"{provider} {prefix} ({response.status_code})"
                + (f": {detail}" if detail else "")
            )

        try:
            return response.json()
        except ValueError as exc:
            raise LLMProviderError(f"{provider} returned invalid JSON") from exc


class _OpenAIAdapter(_BaseAdapter):
    def list_models(self, config: LLMConfig) -> list[ModelSummary]:
        data = self._request_json(
            "openai",
            "GET",
            OPENAI_MODELS_URL,
            headers=_bearer_headers(config.api_key),
            timeout=10.0,
        )
        models = [
            ModelSummary(id=model["id"])
            for model in data.get("data", [])
            if model.get("id")
        ]
        return models

    def generate_text(self, config: LLMConfig, messages: list[dict]) -> LLMResult:
        payload = {
            "model": config.model,
            "stream": False,
            "messages": _coerce_openai_messages(messages),
        }
        data = self._request_json(
            "openai",
            "POST",
            OPENAI_CHAT_URL,
            headers=_bearer_headers(config.api_key),
            json_body=payload,
            timeout=120.0,
        )
        choices = data.get("choices", [])
        if not choices:
            raise LLMProviderError("openai returned no choices")
        message = choices[0].get("message", {})
        text = _flatten_openai_content(message.get("content"))
        if not text:
            raise LLMProviderError("openai returned an empty message")
        return LLMResult(text=text, provider="openai", model=config.model)


class _AnthropicAdapter(_BaseAdapter):
    def list_models(self, config: LLMConfig) -> list[ModelSummary]:
        data = self._request_json(
            "anthropic",
            "GET",
            ANTHROPIC_MODELS_URL,
            headers=_anthropic_headers(config.api_key),
            timeout=10.0,
        )
        models = [
            ModelSummary(id=model["id"], display_name=model.get("display_name"))
            for model in data.get("data", [])
            if model.get("id")
        ]
        return models

    def generate_text(self, config: LLMConfig, messages: list[dict]) -> LLMResult:
        system_text, conversation = _split_system_messages(messages)
        payload: dict[str, object] = {
            "model": config.model,
            "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
            "messages": [
                {"role": item["role"], "content": item["content"]}
                for item in conversation
            ],
        }
        if system_text:
            payload["system"] = system_text
        data = self._request_json(
            "anthropic",
            "POST",
            ANTHROPIC_MESSAGES_URL,
            headers=_anthropic_headers(config.api_key),
            json_body=payload,
            timeout=120.0,
        )
        text = _flatten_anthropic_content(data.get("content"))
        if not text:
            raise LLMProviderError("anthropic returned an empty message")
        return LLMResult(text=text, provider="anthropic", model=config.model)


class _GeminiAdapter(_BaseAdapter):
    def list_models(self, config: LLMConfig) -> list[ModelSummary]:
        data = self._request_json(
            "gemini",
            "GET",
            GEMINI_MODELS_URL,
            headers=_gemini_headers(config.api_key),
            timeout=10.0,
        )
        seen: set[str] = set()
        models: list[ModelSummary] = []
        for model in data.get("models", []):
            methods = model.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            model_id = model.get("baseModelId") or _normalize_gemini_model_name(
                model.get("name", "")
            )
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            models.append(
                ModelSummary(id=model_id, display_name=model.get("displayName"))
            )
        return models

    def generate_text(self, config: LLMConfig, messages: list[dict]) -> LLMResult:
        system_text, conversation = _split_system_messages(messages)
        payload: dict[str, object] = {
            "contents": [
                {
                    "role": "model" if item["role"] == "assistant" else "user",
                    "parts": [{"text": item["content"]}],
                }
                for item in conversation
            ],
            "generationConfig": {"maxOutputTokens": DEFAULT_MAX_OUTPUT_TOKENS},
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        data = self._request_json(
            "gemini",
            "POST",
            _gemini_generate_url(config.model),
            headers=_gemini_headers(config.api_key),
            json_body=payload,
            timeout=120.0,
        )
        candidates = data.get("candidates", [])
        if not candidates:
            block_reason = data.get("promptFeedback", {}).get("blockReason")
            if block_reason:
                raise LLMProviderError(f"gemini blocked the prompt: {block_reason}")
            raise LLMProviderError("gemini returned no candidates")
        text = _flatten_gemini_content(candidates[0].get("content", {}).get("parts"))
        if not text:
            raise LLMProviderError("gemini returned an empty candidate")
        return LLMResult(text=text, provider="gemini", model=config.model)


def list_models(
    config: LLMConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[ModelSummary]:
    adapter = _adapter_for(config.provider, transport=transport)
    return adapter.list_models(config)


def generate_text(
    config: LLMConfig,
    messages: list[dict],
    *,
    transport: httpx.BaseTransport | None = None,
) -> LLMResult:
    adapter = _adapter_for(config.provider, transport=transport)
    return adapter.generate_text(config, messages)


def _adapter_for(
    provider: ProviderKind,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ProviderAdapter:
    if provider == "openai":
        return _OpenAIAdapter(transport=transport)
    if provider == "anthropic":
        return _AnthropicAdapter(transport=transport)
    if provider == "gemini":
        return _GeminiAdapter(transport=transport)
    raise LLMProviderError(f"unsupported provider: {provider}")


def _bearer_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _gemini_headers(api_key: str) -> dict[str, str]:
    return {
        "x-goog-api-key": api_key,
        "content-type": "application/json",
    }


def _coerce_openai_messages(messages: list[dict]) -> list[dict[str, str]]:
    return [
        {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
        }
        for message in messages
    ]


def _split_system_messages(messages: list[dict]) -> tuple[str, list[dict[str, str]]]:
    system_chunks: list[str] = []
    conversation: list[dict[str, str]] = []

    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_chunks.append(content)
            continue
        conversation.append({"role": role, "content": content})

    return "\n\n".join(system_chunks).strip(), conversation


def _flatten_openai_content(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_value = item.get("text")
                if isinstance(text_value, str):
                    texts.append(text_value.strip())
        return "\n".join(text for text in texts if text).strip()
    return ""


def _flatten_anthropic_content(content: object) -> str:
    if not isinstance(content, list):
        return ""
    texts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            text_value = item.get("text")
            if isinstance(text_value, str):
                texts.append(text_value.strip())
    return "\n".join(text for text in texts if text).strip()


def _flatten_gemini_content(parts: object) -> str:
    if not isinstance(parts, list):
        return ""
    texts = []
    for part in parts:
        if isinstance(part, dict):
            text_value = part.get("text")
            if isinstance(text_value, str):
                texts.append(text_value.strip())
    return "\n".join(text for text in texts if text).strip()


def _gemini_generate_url(model: str) -> str:
    model_name = _normalize_gemini_model_name(model)
    return f"{GEMINI_BASE_URL}/models/{model_name}:generateContent"


def _normalize_gemini_model_name(model: str) -> str:
    if model.startswith("models/"):
        return model.split("/", 1)[1]
    return model


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text.strip()

    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message.strip()
    if isinstance(error, str):
        return error.strip()
    message = data.get("message")
    if isinstance(message, str):
        return message.strip()
    return response.text.strip()
