from __future__ import annotations

import unittest

from cli.screens.setup_screen import (
    PROVIDER_LABELS,
    build_llm_config,
    model_summaries_to_options,
)
from llm import ModelSummary


class SetupScreenHelperTests(unittest.TestCase):
    def test_supported_provider_labels_only_include_hosted_providers(self) -> None:
        self.assertEqual(set(PROVIDER_LABELS), {"openai", "anthropic", "gemini"})

    def test_build_llm_config_accepts_manual_model_entry(self) -> None:
        config = build_llm_config("openai", "sk-test", "gpt-manual")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.model, "gpt-manual")

    def test_build_llm_config_rejects_missing_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "api key required"):
            build_llm_config("gemini", "", "gemini-test")

        with self.assertRaisesRegex(ValueError, "model required"):
            build_llm_config("anthropic", "sk-test", "")

    def test_model_summaries_to_options_formats_display_names(self) -> None:
        options = model_summaries_to_options(
            [
                ModelSummary(id="model-a", display_name="Model A"),
                ModelSummary(id="model-b", display_name=None),
            ]
        )

        self.assertEqual(
            options,
            [
                ("Model A (model-a)", "model-a"),
                ("model-b", "model-b"),
            ],
        )
