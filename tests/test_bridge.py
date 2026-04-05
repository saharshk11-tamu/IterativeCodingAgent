from __future__ import annotations

import unittest
from unittest.mock import patch

from cli.bridge import AgentBridge
from cli.events import AgentActivity, AgentDone
from llm import LLMConfig, LLMResult


class _FakeApp:
    def call_from_thread(self, callback, message) -> None:
        callback(message)


class _FakeScreen:
    def __init__(self) -> None:
        self.app = _FakeApp()
        self.messages = []

    def post_message(self, message) -> None:
        self.messages.append(message)


class AgentBridgeTests(unittest.TestCase):
    def test_call_llm_uses_shared_provider_backend(self) -> None:
        config = LLMConfig(provider="openai", model="gpt-test", api_key="sk-test")
        bridge = AgentBridge(config, _FakeScreen())
        messages = [{"role": "user", "content": "Hello"}]

        with patch(
            "cli.bridge.generate_text",
            return_value=LLMResult(
                text="Hi",
                provider="openai",
                model="gpt-test",
            ),
        ) as mock_generate:
            result = bridge.call_llm(messages)

        self.assertEqual(result, "Hi")
        mock_generate.assert_called_once_with(config, messages)

    def test_run_posts_provider_errors_to_activity_pane(self) -> None:
        config = LLMConfig(provider="openai", model="gpt-test", api_key="sk-test")
        screen = _FakeScreen()
        bridge = AgentBridge(config, screen)

        with patch(
            "agent.intake.IntakeAgent.run",
            side_effect=RuntimeError("openai authentication failed"),
        ):
            bridge._run_intake("hello")

        self.assertTrue(
            any(
                isinstance(message, AgentActivity)
                and message.kind == "error"
                and message.text == "openai authentication failed"
                for message in screen.messages
            )
        )
        self.assertTrue(any(isinstance(message, AgentDone) for message in screen.messages))
