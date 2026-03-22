"""
cli/widgets/prompt_pane.py — Left 60% pane

Two input modes:
  normal  — user starts a new task  → posts UserSubmitted
  reply   — user answers agent question → posts UserReplied

Key behaviour:
  Enter        — submit
  Shift+Enter  — insert newline
"""

from __future__ import annotations

from rich.markdown import Markdown as RichMarkdown
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import RichLog, Static, TextArea

_AGENT_PREFIX = "[bold #a6e22e]agent[/bold #a6e22e]"

from cli.events import UserReplied, UserSubmitted


class PromptInput(TextArea):
    """TextArea that submits on Enter and inserts newline on Shift+Enter."""

    class Submit(Message):
        def __init__(self, text_area: "PromptInput") -> None:
            super().__init__()
            self.text_area = text_area

    async def _on_key(self, event) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submit(self))


class PromptPane(Vertical):
    """Left conversation pane."""

    _reply_mode: bool = False
    _stream_buffer: str = ""

    def compose(self) -> ComposeResult:
        yield RichLog(id="history", wrap=True, markup=True, highlight=True)
        yield Static("", id="streaming", markup=True)
        yield Static("enter to send", id="input-hint")
        yield Vertical(
            PromptInput(id="prompt-input", language=None),
            id="input-row",
        )

    def focus_input(self) -> None:
        self.query_one("#prompt-input", PromptInput).focus()

    # ── Submit ────────────────────────────────────────────────────────────────

    def on_prompt_input_submit(self, event: PromptInput.Submit) -> None:
        ta = event.text_area
        text = ta.text.strip()
        if not text:
            return

        log = self.query_one("#history", RichLog)

        if self._reply_mode:
            log.write(f"[bold #66d9ef]you[/bold #66d9ef]  {text}\n")
            ta.clear()
            self._exit_reply_mode()
            self.post_message(UserReplied(text))
        else:
            log.write(f"[bold #66d9ef]you[/bold #66d9ef]  {text}\n")
            ta.clear()
            self.post_message(UserSubmitted(text))

    # ── Clarification question from agent ─────────────────────────────────────

    def show_question(self, question: str) -> None:
        log = self.query_one("#history", RichLog)
        log.write(f"\n[bold #e6db74]agent asks[/bold #e6db74]  {question}\n")
        self._enter_reply_mode()

    def _enter_reply_mode(self) -> None:
        self._reply_mode = True
        hint = self.query_one("#input-hint", Static)
        hint.update("answering agent question  ·  enter to reply")
        hint.add_class("reply-mode")
        self.query_one("#prompt-input", PromptInput).focus()

    def _exit_reply_mode(self) -> None:
        self._reply_mode = False
        hint = self.query_one("#input-hint", Static)
        hint.update("enter to send")
        hint.remove_class("reply-mode")

    # ── Streaming helpers ─────────────────────────────────────────────────────

    def append_token(self, token: str) -> None:
        self._stream_buffer += token
        self.query_one("#streaming", Static).update(_AGENT_PREFIX + "  " + self._stream_buffer)

    def finalize_response(self) -> None:
        if self._stream_buffer:
            log = self.query_one("#history", RichLog)
            log.write(_AGENT_PREFIX)
            log.write(RichMarkdown(self._stream_buffer))
            log.write("")
            self._stream_buffer = ""
            self.query_one("#streaming", Static).update("")

    def clear(self) -> None:
        self._exit_reply_mode()
        self._stream_buffer = ""
        self.query_one("#streaming", Static).update("")
        self.query_one("#history", RichLog).clear()
