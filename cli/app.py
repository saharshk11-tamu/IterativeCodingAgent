"""
cli/app.py — Textual App entry point
"""

from textual.app import App
from textual.binding import Binding

from cli.screens.chat_screen import ChatScreen
from cli.screens.setup_screen import SetupScreen


class AgentApp(App):
    """Main Textual application."""

    CSS_PATH = "theme.tcss"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+s", "push_screen('setup')", "Settings", show=True),
    ]

    SCREENS = {
        "setup": SetupScreen,
    }

    def on_mount(self) -> None:
        self.theme = "monokai"
        self.push_screen("setup")

    def on_setup_screen_connected(self, event) -> None:
        """Fired when the user completes LLM setup."""
        self.push_screen(ChatScreen(llm_config=event.config))


def run():
    app = AgentApp()
    app.run()


if __name__ == "__main__":
    run()
