"""
cli/widgets/agent_pane.py — Right 40% pane

Displays agent activity: tool calls, thinking steps, status updates.
Each entry is timestamped and color-coded by kind.
"""

from __future__ import annotations

from datetime import datetime

from rich.markdown import Markdown as RichMarkdown
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog

# Color map for activity kinds
KIND_STYLES: dict[str, str] = {
    "thinking":    "bold #e6db74",   # monokai yellow
    "tool_call":   "bold #ae81ff",   # monokai purple
    "tool_result": "#a6e22e",        # monokai green
    "status":      "bold #66d9ef",   # monokai cyan
    "error":       "bold #f92672",   # monokai red
    "done":        "bold #a6e22e",   # monokai green
    "task_ready":  "bold #fd971f",   # monokai orange
}

# Kinds that render their body as Markdown
_MARKDOWN_KINDS = {"thinking", "task_ready"}


class AgentPane(Vertical):
    """Right agent activity pane."""

    DEFAULT_CSS = """
    AgentPane {
        width: 40%;
    }
    #activity-log {
        height: 1fr;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield RichLog(id="activity-log", wrap=True, markup=True, highlight=True)

    def add_activity(self, kind: str, text: str) -> None:
        """Add a timestamped activity entry."""
        log = self.query_one("#activity-log", RichLog)
        ts = datetime.now().strftime("%H:%M:%S")
        style = KIND_STYLES.get(kind, "white")
        label = kind.replace("_", " ")
        if kind in _MARKDOWN_KINDS:
            log.write(f"[dim]{ts}[/dim]  [{style}]{label}[/]")
            try:
                log.write(RichMarkdown(text))
            except Exception as exc:
                log.write(f"[dim](render error: {exc})[/dim]")
                log.write(text)
        else:
            log.write(f"[dim]{ts}[/dim]  [{style}]{label}[/]  {text}")

    def clear(self) -> None:
        self.query_one("#activity-log", RichLog).clear()
