"""
cli/screens/setup_screen.py - LLM connection screen.

Supported providers:
  - OpenAI
  - Anthropic
  - Gemini
  - Ollama

Keyboard shortcuts
------------------
  Tab / Shift+Tab  - move between fields
  Arrow keys       - switch provider (inside the RadioSet)
  Enter            - on API key: fetch models for hosted providers; on model field: connect
  F5               - fetch models
  Ctrl+S           - connect
  Escape           - quit
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import (
    Footer,
    Input,
    Label,
    LoadingIndicator,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

from llm import LLMConfig, ModelSummary, OLLAMA_BASE_URL, ProviderKind, list_models

PROVIDER_LABELS: dict[ProviderKind, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "ollama": "Ollama",
}

MODEL_PLACEHOLDERS: dict[ProviderKind, str] = {
    "openai": "Enter OpenAI model ID",
    "anthropic": "Enter Anthropic model ID",
    "gemini": "Enter Gemini model ID",
    "ollama": "Enter Ollama model ID (for example gemma4:latest)",
}

API_KEY_PLACEHOLDERS: dict[ProviderKind, str] = {
    "openai": "OpenAI API key",
    "anthropic": "Anthropic API key",
    "gemini": "Gemini API key",
    "ollama": "",
}


def provider_requires_api_key(provider: ProviderKind) -> bool:
    return provider != "ollama"


def build_llm_config(
    provider: ProviderKind,
    api_key: str,
    model: str,
    *,
    require_model: bool = True,
) -> LLMConfig:
    api_key = api_key.strip()
    model = model.strip()

    if provider_requires_api_key(provider) and not api_key:
        raise ValueError("api key required")
    if require_model and not model:
        raise ValueError("model required")
    if provider == "ollama":
        api_key = api_key or "ollama"

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=OLLAMA_BASE_URL if provider == "ollama" else None,
    )


def model_summaries_to_options(
    models: list[ModelSummary],
) -> list[tuple[str, str]]:
    return [
        (
            f"{model.display_name} ({model.id})"
            if model.display_name and model.display_name != model.id
            else model.id,
            model.id,
        )
        for model in models
    ]


class SetupScreen(Screen):
    """Connection setup screen shown on launch."""

    BINDINGS = [
        Binding("escape", "app.quit", "Quit"),
        Binding("ctrl+s", "connect", "Connect"),
        Binding("f5", "fetch_models", "Fetch models"),
    ]

    class Connected(Message):
        def __init__(self, config: LLMConfig) -> None:
            super().__init__()
            self.config = config

    provider: reactive[ProviderKind] = reactive("openai")

    def compose(self) -> ComposeResult:
        yield Container(
            Static("agent setup", id="setup-title"),
            Static("connect an llm provider to get started", id="setup-subtitle"),
            RadioSet(
                RadioButton("openai", value=True, id="radio-openai"),
                RadioButton("anthropic", id="radio-anthropic"),
                RadioButton("gemini", id="radio-gemini"),
                RadioButton("ollama", id="radio-ollama"),
                id="provider-toggle",
            ),
            Vertical(
                Label("api key  [dim](Enter to fetch models)[/dim]", id="api-key-label"),
                Input(password=True, id="api-key"),
                Label(
                    "model  [dim](manual entry always allowed)[/dim]",
                    id="model-label",
                ),
                Input(id="model-input"),
                Label(
                    "available models  [dim](optional shortcut after F5)[/dim]",
                    id="model-select-label",
                ),
                Select([], prompt="press F5 to fetch models...", id="model-select"),
                id="panel-hosted",
            ),
            Static("", id="status-line"),
            LoadingIndicator(id="spinner"),
            id="setup-box",
        )
        yield Footer()

    def on_mount(self) -> None:
        self._apply_provider_hints()
        self.query_one("#provider-toggle", RadioSet).focus()

    @on(RadioSet.Changed, "#provider-toggle")
    def _provider_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed.id == "radio-anthropic":
            self.provider = "anthropic"
        elif event.pressed.id == "radio-gemini":
            self.provider = "gemini"
        elif event.pressed.id == "radio-ollama":
            self.provider = "ollama"
        else:
            self.provider = "openai"

        self._apply_provider_hints()
        if provider_requires_api_key(self.provider):
            self.query_one("#api-key", Input).focus()
        else:
            self.query_one("#model-input", Input).focus()

    @on(Input.Submitted, "#api-key")
    def _api_key_submitted(self) -> None:
        self.action_fetch_models()

    @on(Input.Submitted, "#model-input")
    def _model_submitted(self) -> None:
        self.action_connect()

    @on(Select.Changed, "#model-select")
    def _model_selected(self, event: Select.Changed) -> None:
        if not isinstance(event.value, str) or not event.value:
            return
        model_input = self.query_one("#model-input", Input)
        model_input.value = event.value
        model_input.focus()

    def action_fetch_models(self) -> None:
        api_key = self.query_one("#api-key", Input).value.strip()
        if provider_requires_api_key(self.provider) and not api_key:
            self._set_status("api key required to fetch models", error=True)
            return
        config = build_llm_config(
            self.provider,
            api_key,
            "",
            require_model=False,
        )
        self._fetch_models(config)

    def action_connect(self) -> None:
        try:
            config = build_llm_config(
                self.provider,
                self.query_one("#api-key", Input).value,
                self.query_one("#model-input", Input).value,
            )
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return

        self._on_connected(config)

    @work(exclusive=True, thread=True)
    def _fetch_models(self, config: LLMConfig) -> None:
        provider_label = PROVIDER_LABELS[config.provider].lower()
        self.app.call_from_thread(self._set_status, f"fetching {provider_label} models...")

        try:
            models = list_models(config)
        except Exception as exc:
            self.app.call_from_thread(
                self._set_status,
                f"error: {exc}",
                True,
            )
            return

        if not models:
            self.app.call_from_thread(
                self._set_status,
                "no models returned; manual model entry is still available",
                True,
            )
            return

        self.app.call_from_thread(self._update_model_select, models)
        self.app.call_from_thread(
            self._set_status,
            f"found {len(models)} model(s) - select one or enter a model id manually",
        )

    def _apply_provider_hints(self) -> None:
        api_key_label = self.query_one("#api-key-label", Label)
        api_key_input = self.query_one("#api-key", Input)
        model_label = self.query_one("#model-label", Label)
        model_input = self.query_one("#model-input", Input)
        model_select = self.query_one("#model-select", Select)

        show_api_key = provider_requires_api_key(self.provider)
        api_key_label.display = show_api_key
        api_key_input.display = show_api_key
        api_key_input.disabled = not show_api_key
        api_key_input.placeholder = API_KEY_PLACEHOLDERS[self.provider]
        model_input.placeholder = MODEL_PLACEHOLDERS[self.provider]
        model_select.set_options([])

        if self.provider == "ollama":
            model_label.update("model  [dim](manual entry or F5 for local models)[/dim]")
            model_select.prompt = "press F5 to query local Ollama models..."
            self._set_status("Ollama selected - local API key not required")
            return

        model_label.update("model  [dim](manual entry always allowed)[/dim]")
        model_select.prompt = "press F5 to fetch models..."
        self._set_status(f"{PROVIDER_LABELS[self.provider]} selected")

    def _update_model_select(self, models: list[ModelSummary]) -> None:
        select = self.query_one("#model-select", Select)
        select.set_options(model_summaries_to_options(models))
        select.focus()

    def _on_connected(self, config: LLMConfig) -> None:
        provider_label = PROVIDER_LABELS[config.provider]
        self._set_status(f"connected · {provider_label} · {config.model}")
        self.post_message(self.Connected(config))

    def _set_status(self, msg: str, error: bool = False) -> None:
        line = self.query_one("#status-line", Static)
        line.update(msg)
        line.remove_class("error")
        if error:
            line.add_class("error")
