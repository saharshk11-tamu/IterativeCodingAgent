"""
cli/bridge.py - Async bridge between Textual UI and hosted LLM providers.

Clarification flow
------------------
The agent calls ask_user(question) when it needs more context:

  1. ask_user() posts AgentQuestion to the UI -> input box enters reply mode
  2. ask_user() calls reply_queue.get()       -> this thread blocks
  3. User types an answer and hits Enter
  4. ChatScreen catches UserReplied           -> calls bridge.reply_queue.put(text)
  5. reply_queue.get() unblocks               -> ask_user() returns the answer
  6. Agent continues with the answer in hand
"""

from __future__ import annotations

import logging
import queue
import threading

from textual.screen import Screen

from cli.events import AgentActivity, AgentDone, AgentMessage, AgentQuestion, AgentToken
from llm import LLMConfig, generate_text

log = logging.getLogger("bridge")
logging.basicConfig(
    filename="bridge-debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


class AgentBridge:
    """
    Wraps hosted LLM access and exposes ask_user() for mid-run clarification.

    Usage (from ChatScreen):
        self.bridge = AgentBridge(self.llm_config, self)
        self.bridge.run(user_text)
    """

    def __init__(self, config: LLMConfig, screen: Screen) -> None:
        self.config = config
        self.screen = screen
        self.reply_queue: queue.Queue[str] = queue.Queue()

    def run(self, user_text: str) -> None:
        threading.Thread(
            target=self._run_intake, args=(user_text,), daemon=True
        ).start()

    def ask_user(self, question: str) -> str:
        """Post a question to the UI and block until the user answers."""
        self._post_question(question)
        answer = self.reply_queue.get()
        self._post_activity("status", "got answer, resuming...")
        return answer

    def post_activity(self, kind: str, text: str) -> None:
        self._post_activity(kind, text)

    def post_token(self, token: str) -> None:
        self._post_token(token)

    def post_message(self, text: str) -> None:
        self._post_message(text)

    def call_llm(self, messages: list[dict]) -> str:
        """Synchronous LLM call used by IntakeAgent for classification/extraction."""
        return generate_text(self.config, messages).text

    def _run_intake(self, user_text: str) -> None:
        from agent.intake import IntakeAgent
        from agent.test_generator import TestGenerator

        try:
            intake = IntakeAgent(self)
            spec = intake.run(user_text)
            if spec is not None:
                self._post_activity("task_ready", _format_spec_summary(spec))
                self._post_activity("status", "transitioning to test generation...")

                test_gen = TestGenerator(self)
                test_gen.generate_and_save(spec)

            # TODO: pass test_code and spec to the next pipeline stage.

        except Exception as exc:
            log.exception("Agent run failed")
            self._post_activity("error", str(exc))
        finally:
            self._post_done()

    def _post_token(self, token: str) -> None:
        self.screen.app.call_from_thread(self.screen.post_message, AgentToken(token))

    def _post_message(self, text: str) -> None:
        self.screen.app.call_from_thread(self.screen.post_message, AgentMessage(text))

    def _post_activity(self, kind: str, text: str) -> None:
        self.screen.app.call_from_thread(
            self.screen.post_message, AgentActivity(kind, text)
        )

    def _post_question(self, question: str) -> None:
        self.screen.app.call_from_thread(
            self.screen.post_message, AgentQuestion(question)
        )

    def _post_done(self) -> None:
        self.screen.app.call_from_thread(self.screen.post_message, AgentDone())


def _format_spec_summary(spec) -> str:
    lines = [
        "**Task ready**\n",
        f"- **Language**: {spec.language}",
        f"- **Type**: {spec.task_type}",
        f"- **Description**: {spec.refined_description}",
    ]
    if spec.requirements:
        lines.append("- **Requirements**:")
        lines.extend(f"  - {requirement}" for requirement in spec.requirements)
    if spec.constraints:
        lines.append("- **Constraints**:")
        lines.extend(f"  - {constraint}" for constraint in spec.constraints)
    if spec.dependencies:
        lines.append(f"- **Dependencies**: {', '.join(spec.dependencies)}")
    if spec.success_metrics:
        lines.append("- **Success metrics**:")
        lines.extend(f"  - {metric}" for metric in spec.success_metrics)
    return "\n".join(lines)
