"""
cli/screens/setup_screen.py — LLM connection screen

Two connection modes:
  - Ollama:    local, model list fetched from Ollama API
  - TAMU LLM: remote, requires API key + https://chat-api.tamu.ai/chat

Keyboard shortcuts
------------------
  Tab / Shift+Tab  — move between fields
  Arrow keys       — switch provider (inside the RadioSet)
  Enter            — on host field: fetch models; on last field: connect
  F5               — fetch Ollama models
  Ctrl+S           — connect
  Escape           — quit
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
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

# ── Data ─────────────────────────────────────────────────────────────────────


@dataclass
class LLMConfig:
    provider: str  # "ollama" | "tamu"
    model: str
    api_key: str | None  # None for Ollama
    base_url: str


TAMU_BASE_URL = "https://chat-api.tamu.ai/openai/chat/completions"
OLLAMA_BASE_URL = "http://localhost:11434"


# ── Screen ────────────────────────────────────────────────────────────────────


class SetupScreen(Screen):
    """Connection setup screen shown on launch."""

    BINDINGS = [
        Binding("escape", "app.quit", "Quit"),
        Binding("ctrl+s", "connect", "Connect"),
        Binding("f5", "fetch_models", "Fetch models"),
    ]

    # Published when the user successfully connects
    class Connected(Message):
        def __init__(self, config: LLMConfig) -> None:
            super().__init__()
            self.config = config

    # Internal state
    provider: reactive[str] = reactive("ollama")

    def compose(self) -> ComposeResult:
        yield Container(
            Static("agent setup", id="setup-title"),
            Static("connect an llm to get started", id="setup-subtitle"),
            # Provider selector — arrow keys to switch
            RadioSet(
                RadioButton("ollama", value=True, id="radio-ollama"),
                RadioButton("tamu llm", id="radio-tamu"),
                id="provider-toggle",
            ),
            # ── Ollama panel ─────────────────────────────────────────────
            Vertical(
                Label("ollama host  [dim](Enter to fetch models)[/dim]"),
                Input(
                    value=OLLAMA_BASE_URL,
                    placeholder="http://localhost:11434",
                    id="ollama-host",
                ),
                Label("model"),
                Select(
                    [],
                    prompt="fetch models first  (F5)...",
                    id="ollama-model-select",
                ),
                id="panel-ollama",
            ),
            # ── TAMU panel ───────────────────────────────────────────────
            Vertical(
                Label("api key  [dim](Enter to fetch models)[/dim]"),
                Input(
                    placeholder="sk-...",
                    password=True,
                    id="tamu-api-key",
                ),
                Label("endpoint"),
                Input(
                    value=TAMU_BASE_URL,
                    id="tamu-endpoint",
                ),
                Label("model"),
                Select(
                    [],
                    prompt="enter api key and press Enter...",
                    id="tamu-model-select",
                ),
                id="panel-tamu",
                classes="hidden",
            ),
            # Status + spinner
            Static("", id="status-line"),
            LoadingIndicator(id="spinner"),
            id="setup-box",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#provider-toggle", RadioSet).focus()

    # ── Provider toggle ───────────────────────────────────────────────────────

    @on(RadioSet.Changed, "#provider-toggle")
    def _provider_changed(self, event: RadioSet.Changed) -> None:
        if event.pressed.id == "radio-ollama":
            self.provider = "ollama"
            self.query_one("#panel-ollama").remove_class("hidden")
            self.query_one("#panel-tamu").add_class("hidden")
            self.query_one("#ollama-host", Input).focus()
        else:
            self.provider = "tamu"
            self.query_one("#panel-tamu").remove_class("hidden")
            self.query_one("#panel-ollama").add_class("hidden")
            self.query_one("#tamu-api-key", Input).focus()

    # ── Ollama host submitted → auto-fetch models ─────────────────────────────

    @on(Input.Submitted, "#ollama-host")
    def _host_submitted(self) -> None:
        self.action_fetch_models()

    # ── TAMU api key submitted → fetch models ─────────────────────────────────

    @on(Input.Submitted, "#tamu-api-key")
    def _tamu_key_submitted(self) -> None:
        self.action_fetch_models()

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_fetch_models(self) -> None:
        if self.provider == "ollama":
            host = self.query_one("#ollama-host", Input).value.rstrip("/")
            self._fetch_ollama_models(host)
        else:
            api_key = self.query_one("#tamu-api-key", Input).value.strip()
            if not api_key:
                self._set_status("api key required to fetch models", error=True)
                return
            self._fetch_tamu_models(api_key)

    def action_connect(self) -> None:
        if self.provider == "ollama":
            self._connect_ollama()
        else:
            self._connect_tamu()

    # ── Fetch Ollama models ───────────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _fetch_ollama_models(self, host: str) -> None:
        self.app.call_from_thread(self._set_status, "fetching models...")
        try:
            resp = httpx.get(f"{host}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            self.app.call_from_thread(self._update_model_select, models)
            self.app.call_from_thread(self._set_status, f"found {len(models)} model(s)  —  Ctrl+S to connect")
        except Exception as exc:
            self.app.call_from_thread(self._set_status, f"error: {exc}", error=True)

    def _update_model_select(self, models: list[str]) -> None:
        select = self.query_one("#ollama-model-select", Select)
        select.set_options([(m, m) for m in models])
        select.focus()

    # ── Fetch TAMU models ─────────────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _fetch_tamu_models(self, api_key: str) -> None:
        self.app.call_from_thread(self._set_status, "fetching tamu models...")
        endpoint = self.query_one("#tamu-endpoint", Input).value.strip()
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        models_url = f"{parsed.scheme}://{parsed.netloc}/openai/models"
        try:
            resp = httpx.get(
                models_url,
                headers={"Authorization": f"Bearer {api_key}", "accept": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            if not models:
                self.app.call_from_thread(self._set_status, "no models returned", error=True)
                return
            self.app.call_from_thread(self._update_tamu_model_select, models)
            self.app.call_from_thread(
                self._set_status, f"found {len(models)} model(s)  —  Ctrl+S to connect"
            )
        except Exception as exc:
            self.app.call_from_thread(self._set_status, f"error: {exc}", error=True)

    def _update_tamu_model_select(self, models: list[str]) -> None:
        select = self.query_one("#tamu-model-select", Select)
        select.set_options([(m, m) for m in models])
        select.focus()

    # ── Connect ───────────────────────────────────────────────────────────────

    @work(exclusive=True, thread=True)
    def _connect_ollama(self) -> None:
        self.app.call_from_thread(self._set_status, "connecting...")
        host = self.query_one("#ollama-host", Input).value.rstrip("/")
        select = self.query_one("#ollama-model-select", Select)
        model = str(select.value) if select.value is not None else ""

        if not model:
            self.app.call_from_thread(
                self._set_status, "select a model first  (F5 to fetch)", error=True
            )
            return

        try:
            httpx.get(f"{host}/api/tags", timeout=5).raise_for_status()
            config = LLMConfig(
                provider="ollama",
                model=model,
                api_key=None,
                base_url=host,
            )
            self.app.call_from_thread(self._on_connected, config)
        except Exception as exc:
            self.app.call_from_thread(
                self._set_status, f"connection failed: {exc}", error=True
            )

    @work(exclusive=True, thread=True)
    def _connect_tamu(self) -> None:
        self.app.call_from_thread(self._set_status, "connecting...")
        api_key = self.query_one("#tamu-api-key", Input).value.strip()
        endpoint = self.query_one("#tamu-endpoint", Input).value.strip()
        select = self.query_one("#tamu-model-select", Select)
        model = str(select.value) if select.value is not None else ""

        if not api_key:
            self.app.call_from_thread(self._set_status, "api key required", error=True)
            return
        if not model:
            self.app.call_from_thread(
                self._set_status, "select a model first  (F5 to fetch)", error=True
            )
            return

        config = LLMConfig(
            provider="tamu",
            model=model,
            api_key=api_key,
            base_url=endpoint,
        )
        self.app.call_from_thread(self._on_connected, config)

    def _on_connected(self, config: LLMConfig) -> None:
        self._set_status(f"connected · {config.provider} · {config.model}")
        self.post_message(self.Connected(config))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg: str, error: bool = False) -> None:
        line = self.query_one("#status-line", Static)
        line.update(msg)
        line.remove_class("error")
        if error:
            line.add_class("error")
