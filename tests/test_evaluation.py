from __future__ import annotations

import json
import unittest

from agent.evaluation import (
    MetricValue,
    EvaluationResult,
    compare_results,
    parse_evaluation_result,
)
from agent.task_spec import MetricSpec


class EvaluationParsingTests(unittest.TestCase):
    def test_parse_evaluation_result(self) -> None:
        payload = json.dumps(
            {
                "metrics": [
                    {
                        "name": "accuracy",
                        "kind": "maximize",
                        "value": 0.91,
                        "target": 0.9,
                        "met": True,
                    }
                ],
                "all_targets_met": True,
                "summary": "ok",
            }
        )

        result = parse_evaluation_result(payload)

        self.assertTrue(result.all_targets_met)
        self.assertEqual(result.metrics[0].value, 0.91)


class EvaluationComparisonTests(unittest.TestCase):
    def test_compare_prefers_more_targets_met(self) -> None:
        specs = [
            MetricSpec("tests", "All tests pass", "pass_fail", True, 1),
            MetricSpec("accuracy", "Validation accuracy", "maximize", 0.8, 2),
        ]
        incumbent = EvaluationResult(
            metrics=[
                MetricValue("tests", "pass_fail", False, True, False),
                MetricValue("accuracy", "maximize", 0.9, 0.8, True),
            ],
            all_targets_met=False,
            summary="baseline",
        )
        candidate = EvaluationResult(
            metrics=[
                MetricValue("tests", "pass_fail", True, True, True),
                MetricValue("accuracy", "maximize", 0.82, 0.8, True),
            ],
            all_targets_met=True,
            summary="better",
        )

        self.assertEqual(compare_results(candidate, incumbent, specs), 1)

    def test_compare_uses_priority_order(self) -> None:
        specs = [
            MetricSpec("accuracy", "Validation accuracy", "maximize", 0.8, 1),
            MetricSpec("latency", "Inference latency", "minimize", 0.5, 2),
        ]
        incumbent = EvaluationResult(
            metrics=[
                MetricValue("accuracy", "maximize", 0.84, 0.8, True),
                MetricValue("latency", "minimize", 0.1, 0.5, True),
            ],
            all_targets_met=True,
            summary="baseline",
        )
        candidate = EvaluationResult(
            metrics=[
                MetricValue("accuracy", "maximize", 0.86, 0.8, True),
                MetricValue("latency", "minimize", 0.2, 0.5, True),
            ],
            all_targets_met=True,
            summary="better primary metric",
        )

        self.assertEqual(compare_results(candidate, incumbent, specs), 1)
