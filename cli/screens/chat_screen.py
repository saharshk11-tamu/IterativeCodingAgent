"""
cli/screens/chat_screen.py — Main 60/40 split pane

Left  (60%): user prompt input + conversation history
Right (40%): agent activity log — tool calls, thinking, status
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header, Rule

from cli.bridge import AgentBridge
from cli.events import (
    AgentActivity,
    AgentDone,
    AgentMessage,
    AgentQuestion,
    AgentToken,
    UserReplied,
    UserSubmitted,
)
from cli.screens.setup_screen import LLMConfig
from cli.widgets.agent_pane import AgentPane
from cli.widgets.prompt_pane import PromptPane


class ChatScreen(Screen):
    """60/40 split: left = conversation, right = agent activity."""

    BINDINGS = [
        Binding("ctrl+l", "clear", "Clear", show=True),
        Binding("ctrl+s", "app.push_screen('setup')", "Settings", show=True),
        Binding("ctrl+q", "app.quit", "Quit", show=True),
    ]

    def __init__(self, llm_config: LLMConfig) -> None:
        super().__init__()
        self.llm_config = llm_config
        self._bridge: "AgentBridge | None" = None  # kept alive between clarifications

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Horizontal(
            PromptPane(id="prompt-pane"),
            Rule(orientation="vertical", id="pane-divider"),
            AgentPane(id="agent-pane"),
            id="split",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.llm_config.provider + " · " + self.llm_config.model
        # Focus input on load
        self.query_one(PromptPane).focus_input()

    # ── Route user submissions to the bridge ─────────────────────────────────

    @on(UserSubmitted)
    def handle_submission(self, event: UserSubmitted) -> None:
        from cli.bridge import AgentBridge

        self._bridge = AgentBridge(self.llm_config, self)
        self._bridge.run(event.text)

    @on(UserReplied)
    def handle_reply(self, event: UserReplied) -> None:
        """Feed the user's clarification answer back to the waiting agent thread."""
        if self._bridge is not None:
            self._bridge.reply_queue.put(event.text)

    @on(AgentQuestion)
    def on_question(self, event: AgentQuestion) -> None:
        """Show the question in the conversation and switch input to reply mode."""
        self.query_one("#prompt-pane", PromptPane).show_question(event.question)

    # ── Route agent events to widgets ────────────────────────────────────────

    @on(AgentMessage)
    def on_agent_message(self, event: AgentMessage) -> None:
        self.query_one("#prompt-pane", PromptPane).add_agent_message(event.text)

    @on(AgentToken)
    def on_token(self, event: AgentToken) -> None:
        pane = self.query_one("#prompt-pane", PromptPane)
        pane.append_token(event.token)

    @on(AgentActivity)
    def on_activity(self, event: AgentActivity) -> None:
        self.query_one("#agent-pane", AgentPane).add_activity(event.kind, event.text)

    @on(AgentDone)
    def on_done(self, event: AgentDone) -> None:
        self.query_one("#prompt-pane", PromptPane).finalize_response()
        self.query_one("#agent-pane", AgentPane).add_activity("done", "run complete")

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_clear(self) -> None:
        self.query_one("#prompt-pane", PromptPane).clear()
        self.query_one("#agent-pane", AgentPane).clear()
