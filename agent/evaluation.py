"""Evaluation result types and comparison helpers for the automation loop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent.task_spec import MetricKind, MetricSpec


@dataclass(frozen=True)
class MetricValue:
    name: str
    kind: MetricKind
    value: float | bool
    target: float | bool
    met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "value": self.value,
            "target": self.target,
            "met": self.met,
        }


@dataclass(frozen=True)
class EvaluationResult:
    metrics: list[MetricValue]
    all_targets_met: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "all_targets_met": self.all_targets_met,
            "summary": self.summary,
        }

    def met_count(self) -> int:
        return sum(1 for metric in self.metrics if metric.met)


def parse_evaluation_result(raw: str) -> EvaluationResult:
    data = json.loads(raw)
    metrics_data = data.get("metrics")
    if not isinstance(metrics_data, list):
        raise ValueError("evaluation result is missing a metrics list")

    metrics: list[MetricValue] = []
    for item in metrics_data:
        if not isinstance(item, dict):
            raise ValueError("evaluation metric entries must be objects")
        kind = str(item.get("kind", "pass_fail")).strip().lower()
        if kind not in {"pass_fail", "maximize", "minimize"}:
            raise ValueError(f"invalid metric kind: {kind}")
        metrics.append(
            MetricValue(
                name=str(item.get("name", "metric")).strip() or "metric",
                kind=kind,
                value=_coerce_metric_value(kind, item.get("value")),
                target=_coerce_metric_value(kind, item.get("target")),
                met=bool(item.get("met", False)),
            )
        )

    return EvaluationResult(
        metrics=metrics,
        all_targets_met=bool(data.get("all_targets_met", False)),
        summary=str(data.get("summary", "")).strip(),
    )


def make_failure_result(
    metric_specs: list[MetricSpec],
    summary: str,
    *,
    default_numeric: float = 0.0,
) -> EvaluationResult:
    metrics = []
    for spec in metric_specs:
        value: float | bool
        if spec.kind == "pass_fail":
            value = False
        elif spec.kind == "maximize":
            value = default_numeric
        else:
            value = max(float(spec.target), default_numeric) + 1.0
        metrics.append(
            MetricValue(
                name=spec.name,
                kind=spec.kind,
                value=value,
                target=spec.target,
                met=False,
            )
        )
    return EvaluationResult(metrics=metrics, all_targets_met=False, summary=summary)


def compare_results(
    candidate: EvaluationResult,
    incumbent: EvaluationResult | None,
    metric_specs: list[MetricSpec],
) -> int:
    """
    Return 1 if candidate is better, -1 if worse, 0 if equivalent.
    """
    if incumbent is None:
        return 1

    if candidate.met_count() != incumbent.met_count():
        return 1 if candidate.met_count() > incumbent.met_count() else -1

    candidate_map = {metric.name: metric for metric in candidate.metrics}
    incumbent_map = {metric.name: metric for metric in incumbent.metrics}

    for spec in sorted(metric_specs, key=lambda metric: metric.priority):
        cand = candidate_map.get(spec.name)
        prev = incumbent_map.get(spec.name)
        if cand is None and prev is None:
            continue
        if cand is None:
            return -1
        if prev is None:
            return 1
        if spec.kind == "pass_fail":
            cand_score = 1 if bool(cand.met) else 0
            prev_score = 1 if bool(prev.met) else 0
        elif spec.kind == "maximize":
            cand_score = float(cand.value)
            prev_score = float(prev.value)
        else:
            cand_score = -float(cand.value)
            prev_score = -float(prev.value)
        if cand_score != prev_score:
            return 1 if cand_score > prev_score else -1

    return 0


def format_evaluation_table(result: EvaluationResult) -> str:
    if not result.metrics:
        return "No metrics reported."
    lines = []
    for metric in result.metrics:
        lines.append(
            f"- {metric.name}: value={metric.value} target={metric.target} met={metric.met}"
        )
    if result.summary:
        lines.append(f"- summary: {result.summary}")
    return "\n".join(lines)


def _coerce_metric_value(kind: MetricKind, value: Any) -> float | bool:
    if kind == "pass_fail":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1", "pass"}
        return bool(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
