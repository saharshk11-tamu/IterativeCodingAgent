"""
cli/bridge.py — Async bridge between Textual UI and the LLM

Handles:
  - Ollama:    POST /api/chat  (streaming NDJSON)
  - TAMU LLM:  POST to chat-api.tamu.ai (OpenAI-compatible SSE)

Clarification flow
------------------
The agent calls ask_user(question) when it needs more context:

  1. ask_user() posts AgentQuestion to the UI  →  input box enters reply mode
  2. ask_user() calls reply_queue.get()        →  THIS THREAD BLOCKS
  3. User types an answer and hits Enter
  4. ChatScreen catches UserReplied            →  calls bridge.reply_queue.put(text)
  5. reply_queue.get() unblocks                →  ask_user() returns the answer
  6. Agent continues with the answer in hand

The UI thread is never blocked — only the agent worker thread waits.

Agent integration
-----------------
AgentBridge satisfies agent.intake.BridgeProtocol, so IntakeAgent can call:
  bridge.ask_user(q)          — clarification (blocking)
  bridge.post_activity(k, t)  — log to agent pane
  bridge.post_token(t)        — stream text to conversation pane
  bridge.call_llm(messages)   — synchronous LLM call, returns str
"""

from __future__ import annotations

import logging
import queue
import threading

import httpx
from textual.screen import Screen

from cli.events import AgentActivity, AgentDone, AgentMessage, AgentQuestion, AgentToken
from cli.screens.setup_screen import LLMConfig

log = logging.getLogger("bridge")
logging.basicConfig(
    filename="bridge-debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


class AgentBridge:
    """
    Wraps LLM streaming and exposes ask_user() for mid-run clarification.

    Usage (from ChatScreen):
        self.bridge = AgentBridge(self.llm_config, self)
        self.bridge.run(user_text)

    When UserReplied fires on the screen:
        self.bridge.reply_queue.put(event.text)
    """

    def __init__(self, config: LLMConfig, screen: Screen) -> None:
        self.config = config
        self.screen = screen
        # Unbounded queue; in practice at most one item sits here at a time
        self.reply_queue: queue.Queue[str] = queue.Queue()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, user_text: str) -> None:
        threading.Thread(
            target=self._run_intake, args=(user_text,), daemon=True
        ).start()

    # ── Clarification helper (called from inside agent worker thread) ─────────

    def ask_user(self, question: str) -> str:
        """
        Post a question to the UI and block until the user answers.

        Call this from within the agent logic (inside a @work thread).
        Returns the user's answer as a plain string.
        """
        self._post_question(question)  # UI shows question, enables reply mode
        answer = self.reply_queue.get()  # blocks this worker thread only
        self._post_activity("status", "got answer, resuming...")
        return answer

    # ── BridgeProtocol public surface (called by IntakeAgent) ────────────────

    def post_activity(self, kind: str, text: str) -> None:
        self._post_activity(kind, text)

    def post_token(self, token: str) -> None:
        self._post_token(token)

    def post_message(self, text: str) -> None:
        self._post_message(text)

    def call_llm(self, messages: list[dict]) -> str:
        """Synchronous LLM call used by IntakeAgent for classification/extraction."""
        if self.config.provider == "ollama":
            return self._call_llm_ollama(messages)
        return self._call_llm_tamu(messages)

    # ── Intake entry point ────────────────────────────────────────────────────

    def _run_intake(self, user_text: str) -> None:
        from agent.intake import IntakeAgent
        from agent.test_generator import TestGenerator

        try:
            intake = IntakeAgent(self)
            spec = intake.run(user_text)
            if spec is not None:
                # 1. Output the intake summary to the agent pane
                self._post_activity("task_ready", _format_spec_summary(spec))

                # 2. Phase: Test Generation
                self._post_activity("status", "transitioning to test generation...")

                test_gen = TestGenerator(self)
                test_gen.generate_and_save(spec)

            # TODO: pass test_code and spec to next pipeline stage (Code Generation / Sandbox execution)

        except Exception as exc:
            self._post_activity("error", str(exc))
        finally:
            self._post_done()

    # ── LLM backends ─────────────────────────────────────────────────────────

    def _call_llm_ollama(self, messages: list[dict]) -> str:
        url = f"{self.config.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": messages,
            "think": True,
        }
        resp = httpx.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        # Prefer the dedicated thinking field; fall back to <think> tags in content
        # Use `or ""` to guard against explicit null values from the API
        thinking = msg.get("thinking") or ""
        content = msg.get("content") or ""
        if not thinking:
            thinking, content = _extract_think_tags(content)
        if thinking:
            self._post_activity("thinking", thinking)
        return content

    def _call_llm_tamu(self, messages: list[dict]) -> str:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.config.model, "stream": False, "messages": messages}
        resp = httpx.post(
            self.config.base_url, json=payload, headers=headers, timeout=120
        )
        resp.raise_for_status()
        log.debug("TAMU raw response: %s", resp.text)
        msg = resp.json().get("choices", [{}])[0].get("message", {})
        # Prefer reasoning_content field; fall back to <think> tags in content
        # Use `or ""` to guard against explicit null values from the API
        thinking = msg.get("reasoning_content") or ""
        content = msg.get("content") or ""
        if not thinking:
            thinking, content = _extract_think_tags(content)
        if thinking:
            self._post_activity("thinking", thinking)
        return content

    # ── Helpers ───────────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

import re as _re


def _extract_think_tags(content: str) -> tuple[str, str]:
    """
    Pull <think>...</think> blocks out of content.
    Returns (thinking_text, cleaned_content).
    """
    thinking_parts: list[str] = []

    def _collect(m: _re.Match) -> str:
        thinking_parts.append(m.group(1).strip())
        return ""

    cleaned = _re.sub(r"<think>(.*?)</think>", _collect, content, flags=_re.DOTALL)
    return "\n\n".join(thinking_parts), cleaned.strip()


def _format_spec_summary(spec) -> str:
    lines = [
        "**Task ready**\n",
        f"- **Language**: {spec.language}",
        f"- **Type**: {spec.task_type}",
        f"- **Description**: {spec.refined_description}",
    ]
    if spec.requirements:
        lines.append("- **Requirements**:")
        lines.extend(f"  - {r}" for r in spec.requirements)
    if spec.constraints:
        lines.append("- **Constraints**:")
        lines.extend(f"  - {c}" for c in spec.constraints)
    if spec.dependencies:
        lines.append(f"- **Dependencies**: {', '.join(spec.dependencies)}")
    if spec.success_metrics:
        lines.append("- **Success metrics**:")
        lines.extend(f"  - {m}" for m in spec.success_metrics)
    return "\n".join(lines)
