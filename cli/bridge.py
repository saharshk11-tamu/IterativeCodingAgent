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
"""

from __future__ import annotations

import json
import queue
import threading

import httpx
from textual.screen import Screen

from cli.events import AgentActivity, AgentDone, AgentQuestion, AgentToken
from cli.screens.setup_screen import LLMConfig


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
        target = (
            self._run_ollama if self.config.provider == "ollama" else self._run_tamu
        )
        threading.Thread(target=target, args=(user_text,), daemon=True).start()

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

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _run_ollama(self, user_text: str) -> None:
        self._post_activity("status", f"sending to {self.config.model}...")
        url = f"{self.config.base_url}/api/chat"
        payload = {
            "model": self.config.model,
            "stream": True,
            "messages": [{"role": "user", "content": user_text}],
        }
        try:
            with httpx.stream("POST", url, json=payload, timeout=60) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("message", {}).get("content", "")
                    if token:
                        self._post_token(token)
                    if data.get("done"):
                        break
            self._post_activity("done", "stream complete")
        except Exception as exc:
            self._post_activity("error", str(exc))
        finally:
            self._post_done()

    # ── TAMU LLM (OpenAI-compatible) ─────────────────────────────────────────

    def _run_tamu(self, user_text: str) -> None:
        self._post_activity("status", f"sending to {self.config.model} via TAMU...")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "stream": False,
            "messages": [{"role": "user", "content": user_text}],
        }
        try:
            resp = httpx.post(
                self.config.base_url,
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                self._post_token(content)
            self._post_activity("done", "response received")
        except Exception as exc:
            self._post_activity("error", str(exc))
        finally:
            self._post_done()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _post_token(self, token: str) -> None:
        self.screen.app.call_from_thread(self.screen.post_message, AgentToken(token))

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
