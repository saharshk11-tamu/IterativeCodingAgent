"""
cli/widgets/agent_pane.py — Right 40% pane

Displays agent activity: tool calls, thinking steps, status updates.
Each entry is timestamped and color-coded by kind.
"""

from __future__ import annotations

from datetime import datetime

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
}


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
        log.write(f"[dim]{ts}[/dim]  [{style}]{label}[/{style}]  {text}")

    def clear(self) -> None:
        self.query_one("#activity-log", RichLog).clear()
