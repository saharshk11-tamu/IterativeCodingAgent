"""
agent/task_spec.py - Structured output of the intake phase.

Passed downstream to evaluator generation and automation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MetricKind = Literal["pass_fail", "maximize", "minimize"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    description: str
    kind: MetricKind
    target: float | bool
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "target": self.target,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], priority: int | None = None) -> "MetricSpec":
        kind = str(data.get("kind", "pass_fail")).strip().lower()
        if kind not in {"pass_fail", "maximize", "minimize"}:
            kind = "pass_fail"

        raw_target = data.get("target", True)
        if kind == "pass_fail":
            if isinstance(raw_target, bool):
                target: float | bool = raw_target
            elif isinstance(raw_target, str):
                target = raw_target.strip().lower() in {"true", "yes", "1", "pass"}
            else:
                target = bool(raw_target)
        else:
            try:
                target = float(raw_target)
            except (TypeError, ValueError):
                target = 0.0

        return cls(
            name=str(data.get("name", "metric")).strip() or "metric",
            description=str(data.get("description", "")).strip() or str(data.get("name", "metric")).strip() or "metric",
            kind=kind,
            target=target,
            priority=priority if priority is not None else int(data.get("priority", 1)),
        )

    def target_text(self) -> str:
        if self.kind == "pass_fail":
            return "must pass" if bool(self.target) else "must fail"
        comparator = ">=" if self.kind == "maximize" else "<="
        return f"{comparator} {self.target}"


@dataclass(frozen=True)
class TaskSpec:
    original_prompt: str
    refined_description: str
    language: str
    task_type: Literal[
        "function",
        "class",
        "script",
        "api_endpoint",
        "data_pipeline",
        "other_coding",
    ]
    requirements: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)
    metrics: list[MetricSpec] = field(default_factory=list)
    clarification_turns: int = 0

    @property
    def success_metrics(self) -> list[str]:
        return [metric.description for metric in self.metrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_prompt": self.original_prompt,
            "refined_description": self.refined_description,
            "language": self.language,
            "task_type": self.task_type,
            "requirements": list(self.requirements),
            "constraints": list(self.constraints),
            "dependencies": list(self.dependencies),
            "examples": list(self.examples),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "clarification_turns": self.clarification_turns,
        }
