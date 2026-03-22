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

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import RichLog, Static, TextArea

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

    DEFAULT_CSS = """
    PromptPane {
        width: 60%;
    }
    #history {
        height: 1fr;
        padding: 1 2;
    }
    #input-row {
        height: auto;
        max-height: 10;
        padding: 0 1 1 1;
    }
    #prompt-input {
        height: auto;
        max-height: 8;
    }
    #input-hint {
        color: $text-muted;
        padding: 0 1;
        height: 1;
    }
    #input-hint.reply-mode {
        color: $warning;
    }
    """

    _reply_mode: bool = False

    def compose(self) -> ComposeResult:
        yield RichLog(id="history", wrap=True, markup=True, highlight=True)
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
        self.query_one("#history", RichLog).write(token)

    def finalize_response(self) -> None:
        self.query_one("#history", RichLog).write("\n")

    def clear(self) -> None:
        self._exit_reply_mode()
        self.query_one("#history", RichLog).clear()
