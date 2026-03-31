"""
agent/task_spec.py — Structured output of the intake phase.

Passed downstream to test generation, code generation, etc.
Frozen so that downstream phases can only read, never mutate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TaskSpec:
    # ── Core description ─────────────────────────────────────────────────────
    original_prompt: str
    # Canonical problem statement after clarification
    refined_description: str

    # ── Scope ────────────────────────────────────────────────────────────────
    language: str  # e.g. "python", "typescript"
    task_type: Literal[
        "function",
        "class",
        "script",
        "api_endpoint",
        "data_pipeline",
        "other_coding",
    ]

    # ── Constraints & context ────────────────────────────────────────────────
    # Functional requirements: "Must return sorted list in ascending order"
    requirements: list[str] = field(default_factory=list)
    # Non-functional constraints: "Must run in O(n log n) time"
    constraints: list[str] = field(default_factory=list)
    # Explicitly mentioned libraries/frameworks only
    dependencies: list[str] = field(default_factory=list)

    # ── Test expectations ────────────────────────────────────────────────────
    # Input/output pairs the user mentioned: [{"input": "...", "output": "..."}]
    examples: list[dict[str, str]] = field(default_factory=list)

    # ── Success metrics ───────────────────────────────────────────────────────
    # Agreed-upon criteria for what "done" looks like, confirmed by the user.
    # e.g. "All unit tests pass", "Handles empty input without raising"
    success_metrics: list[str] = field(default_factory=list)

    # ── Metadata ─────────────────────────────────────────────────────────────
    clarification_turns: int = 0
