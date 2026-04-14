from __future__ import annotations

import json
import unittest

import httpx

from llm.client import (
    ANTHROPIC_MESSAGES_URL,
    ANTHROPIC_MODELS_URL,
    GEMINI_MODELS_URL,
    LLMConfig,
    LLMProviderError,
    OLLAMA_CHAT_URL,
    OLLAMA_TAGS_URL,
    OPENAI_CHAT_URL,
    OPENAI_MODELS_URL,
    generate_text,
    list_models,
)


class OpenAIAdapterTests(unittest.TestCase):
    def test_openai_generate_text_serializes_messages_and_extracts_content(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), OPENAI_CHAT_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer sk-openai")
            payload = json.loads(request.content.decode())
            self.assertEqual(payload["model"], "gpt-test")
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["messages"], messages)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "hello from openai"}},
                    ]
                },
            )

        result = generate_text(
            LLMConfig(provider="openai", model="gpt-test", api_key="sk-openai"),
            messages,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.text, "hello from openai")

    def test_openai_list_models_parses_ids(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url), OPENAI_MODELS_URL)
            return httpx.Response(
                200,
                json={"data": [{"id": "gpt-a"}, {"id": "gpt-b"}]},
            )

        models = list_models(
            LLMConfig(provider="openai", model="", api_key="sk-openai"),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual([model.id for model in models], ["gpt-a", "gpt-b"])

    def test_openai_auth_errors_are_normalized(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"message": "invalid api key"}},
            )

        with self.assertRaisesRegex(
            LLMProviderError,
            r"openai authentication failed \(401\): invalid api key",
        ):
            list_models(
                LLMConfig(provider="openai", model="", api_key="bad-key"),
                transport=httpx.MockTransport(handler),
            )


class OllamaAdapterTests(unittest.TestCase):
    def test_ollama_generate_text_uses_openai_compatible_local_endpoint(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), OLLAMA_CHAT_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer ollama")
            payload = json.loads(request.content.decode())
            self.assertEqual(payload["model"], "gemma4:latest")
            self.assertEqual(payload["messages"], messages)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello from ollama"}}]},
            )

        result = generate_text(
            LLMConfig(
                provider="ollama",
                model="gemma4:latest",
                api_key="ollama",
            ),
            messages,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.text, "hello from ollama")
        self.assertEqual(result.provider, "ollama")

    def test_ollama_generate_text_defaults_dummy_bearer_key_when_blank(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.headers["Authorization"], "Bearer ollama")
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello from ollama"}}]},
            )

        result = generate_text(
            LLMConfig(provider="ollama", model="gemma4:latest", api_key=""),
            [{"role": "user", "content": "Hello"}],
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.text, "hello from ollama")

    def test_ollama_list_models_reads_native_tags_endpoint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "GET")
            self.assertEqual(str(request.url), OLLAMA_TAGS_URL)
            self.assertNotIn("Authorization", request.headers)
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "gemma4:latest"},
                        {"model": "llama3.2"},
                    ]
                },
            )

        models = list_models(
            LLMConfig(provider="ollama", model="", api_key=""),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual([model.id for model in models], ["gemma4:latest", "llama3.2"])


class AnthropicAdapterTests(unittest.TestCase):
    def test_anthropic_generate_text_promotes_system_prompt(self) -> None:
        messages = [
            {"role": "system", "content": "Rule one."},
            {"role": "system", "content": "Rule two."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertEqual(str(request.url), ANTHROPIC_MESSAGES_URL)
            self.assertEqual(request.headers["x-api-key"], "sk-anthropic")
            self.assertEqual(request.headers["anthropic-version"], "2023-06-01")
            payload = json.loads(request.content.decode())
            self.assertEqual(payload["system"], "Rule one.\n\nRule two.")
            self.assertEqual(
                payload["messages"],
                [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ],
            )
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "from anthropic"},
                    ]
                },
            )

        result = generate_text(
            LLMConfig(
                provider="anthropic",
                model="claude-test",
                api_key="sk-anthropic",
            ),
            messages,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.text, "hello\nfrom anthropic")

    def test_anthropic_list_models_parses_ids(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), ANTHROPIC_MODELS_URL)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "claude-a", "display_name": "Claude A"},
                        {"id": "claude-b", "display_name": "Claude B"},
                    ]
                },
            )

        models = list_models(
            LLMConfig(provider="anthropic", model="", api_key="sk-anthropic"),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual([model.id for model in models], ["claude-a", "claude-b"])
        self.assertEqual(models[0].display_name, "Claude A")

    def test_anthropic_empty_content_raises_error(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"content": []})

        with self.assertRaisesRegex(
            LLMProviderError,
            r"anthropic returned an empty message",
        ):
            generate_text(
                LLMConfig(
                    provider="anthropic",
                    model="claude-test",
                    api_key="sk-anthropic",
                ),
                [{"role": "user", "content": "Hi"}],
                transport=httpx.MockTransport(handler),
            )


class GeminiAdapterTests(unittest.TestCase):
    def test_gemini_generate_text_maps_contents_and_system_instruction(self) -> None:
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertIn("/models/gemini-test:generateContent", str(request.url))
            self.assertEqual(request.headers["x-goog-api-key"], "sk-gemini")
            payload = json.loads(request.content.decode())
            self.assertEqual(
                payload["systemInstruction"],
                {"parts": [{"text": "Be concise."}]},
            )
            self.assertEqual(
                payload["contents"],
                [
                    {"role": "user", "parts": [{"text": "Question"}]},
                    {"role": "model", "parts": [{"text": "Previous answer"}]},
                ],
            )
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "hello"},
                                    {"text": "from gemini"},
                                ]
                            }
                        }
                    ]
                },
            )

        result = generate_text(
            LLMConfig(provider="gemini", model="gemini-test", api_key="sk-gemini"),
            messages,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result.text, "hello\nfrom gemini")

    def test_gemini_list_models_filters_generate_content_models(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(str(request.url), GEMINI_MODELS_URL)
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-1",
                            "baseModelId": "gemini-1",
                            "displayName": "Gemini One",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/embed-1",
                            "baseModelId": "embed-1",
                            "displayName": "Embedding One",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                        {
                            "name": "models/gemini-1-001",
                            "baseModelId": "gemini-1",
                            "displayName": "Gemini One Variant",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                    ]
                },
            )

        models = list_models(
            LLMConfig(provider="gemini", model="", api_key="sk-gemini"),
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual([model.id for model in models], ["gemini-1"])
        self.assertEqual(models[0].display_name, "Gemini One")

    def test_gemini_server_errors_are_normalized(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": {"message": "try later"}})

        with self.assertRaisesRegex(
            LLMProviderError,
            r"gemini server error \(500\): try later",
        ):
            generate_text(
                LLMConfig(provider="gemini", model="gemini-test", api_key="sk-gemini"),
                [{"role": "user", "content": "Hi"}],
                transport=httpx.MockTransport(handler),
            )
