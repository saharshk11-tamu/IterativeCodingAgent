"""
cli/screens/setup_screen.py - Hosted LLM connection screen.

Supported providers:
  - OpenAI
  - Anthropic
  - Gemini

Keyboard shortcuts
------------------
  Tab / Shift+Tab  - move between fields
  Arrow keys       - switch provider (inside the RadioSet)
  Enter            - on API key: fetch models; on model field: connect
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

from llm import LLMConfig, ModelSummary, ProviderKind, list_models

PROVIDER_LABELS: dict[ProviderKind, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
}

MODEL_PLACEHOLDERS: dict[ProviderKind, str] = {
    "openai": "Enter OpenAI model ID",
    "anthropic": "Enter Anthropic model ID",
    "gemini": "Enter Gemini model ID",
}

API_KEY_PLACEHOLDERS: dict[ProviderKind, str] = {
    "openai": "OpenAI API key",
    "anthropic": "Anthropic API key",
    "gemini": "Gemini API key",
}


def build_llm_config(provider: ProviderKind, api_key: str, model: str) -> LLMConfig:
    api_key = api_key.strip()
    model = model.strip()

    if not api_key:
        raise ValueError("api key required")
    if not model:
        raise ValueError("model required")

    return LLMConfig(provider=provider, model=model, api_key=api_key)


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
            Static("connect a hosted llm to get started", id="setup-subtitle"),
            RadioSet(
                RadioButton("openai", value=True, id="radio-openai"),
                RadioButton("anthropic", id="radio-anthropic"),
                RadioButton("gemini", id="radio-gemini"),
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
        else:
            self.provider = "openai"

        self._apply_provider_hints()
        self.query_one("#api-key", Input).focus()

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
        if not api_key:
            self._set_status("api key required to fetch models", error=True)
            return
        config = LLMConfig(provider=self.provider, model="", api_key=api_key)
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
        self.query_one("#api-key", Input).placeholder = API_KEY_PLACEHOLDERS[self.provider]
        self.query_one("#model-input", Input).placeholder = MODEL_PLACEHOLDERS[self.provider]
        self.query_one("#model-select", Select).set_options([])
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
