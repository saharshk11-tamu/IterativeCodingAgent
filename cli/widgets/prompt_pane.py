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

import re

from rich.markdown import Markdown as RichMarkdown
from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, RichLog, Static, TextArea

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


# ── Message widgets ───────────────────────────────────────────────────────────

class _UserMessage(Static):
    DEFAULT_CSS = """
    _UserMessage {
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__(
            f"[bold #66d9ef]you[/bold #66d9ef]  {text}",
            markup=True,
        )


class _AgentResponse(Vertical):
    DEFAULT_CSS = """
    _AgentResponse {
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
    }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(f"{_AGENT_PREFIX}", markup=True)
        yield Static(RichMarkdown(self._text))


class _AgentQuestion(Vertical):
    DEFAULT_CSS = """
    _AgentQuestion {
        height: auto;
        padding: 0 2;
        margin-bottom: 1;
    }
    _AgentQuestion Collapsible {
        border: none;
        padding: 0;
        margin-top: 1;
        background: $panel;
    }
    """

    def __init__(self, text: str, code: str = "", lang: str = "") -> None:
        super().__init__()
        self._text = text
        self._code = code
        self._lang = lang

    def compose(self) -> ComposeResult:
        yield Static("[bold #e6db74]agent asks[/bold #e6db74]", markup=True)
        if self._text:
            yield Static(RichMarkdown(self._text))
        if self._code:
            yield Collapsible(
                Static(
                    Syntax(
                        self._code,
                        self._lang or "text",
                        theme="monokai",
                        line_numbers=True,
                    )
                ),
                title="file preview",
                collapsed=True,
            )


# ── Pane ─────────────────────────────────────────────────────────────────────

class PromptPane(Vertical):
    """Left conversation pane."""

    _reply_mode: bool = False
    _stream_buffer: str = ""

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="history")
        yield Static("", id="streaming", markup=True)
        yield Static("enter to send", id="input-hint")
        yield Vertical(
            PromptInput(id="prompt-input", language=None),
            id="input-row",
        )

    def focus_input(self) -> None:
        self.query_one("#prompt-input", PromptInput).focus()

    def _history(self) -> VerticalScroll:
        return self.query_one("#history", VerticalScroll)

    # ── Submit ────────────────────────────────────────────────────────────────

    def on_prompt_input_submit(self, event: PromptInput.Submit) -> None:
        ta = event.text_area
        text = ta.text.strip()
        if not text:
            return

        self._history().mount(_UserMessage(text))
        self._scroll_to_end()
        ta.clear()

        if self._reply_mode:
            self._exit_reply_mode()
            self.post_message(UserReplied(text))
        else:
            self.post_message(UserSubmitted(text))

    # ── Clarification question from agent ─────────────────────────────────────

    def show_question(self, question: str) -> None:
        code_match = re.search(r"```(\w*)\n(.*?)```", question, re.DOTALL)

        if code_match:
            lang = code_match.group(1) or "text"
            code = code_match.group(2)
            text_before = question[: code_match.start()].strip()
            text_after = question[code_match.end() :].strip()
            display_text = "\n\n".join(filter(None, [text_before, text_after]))
            widget = _AgentQuestion(text=display_text, code=code, lang=lang)
        else:
            widget = _AgentQuestion(text=question)

        self._history().mount(widget)
        self._scroll_to_end()
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
        self.query_one("#streaming", Static).update(
            _AGENT_PREFIX + "  " + self._stream_buffer
        )

    def add_agent_message(self, text: str) -> None:
        """Mount a completed agent message directly into the chat history."""
        self._history().mount(_AgentResponse(text))
        self._scroll_to_end()

    def finalize_response(self) -> None:
        if self._stream_buffer:
            self._history().mount(_AgentResponse(self._stream_buffer))
            self._stream_buffer = ""
            self.query_one("#streaming", Static).update("")
            self._scroll_to_end()

    def clear(self) -> None:
        self._exit_reply_mode()
        self._stream_buffer = ""
        self.query_one("#streaming", Static).update("")
        history = self._history()
        for child in list(history.children):
            child.remove()

    def _scroll_to_end(self) -> None:
        self.call_after_refresh(self._history().scroll_end, animate=False)
