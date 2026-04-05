from __future__ import annotations

import unittest

from agent.task_spec import MetricSpec, TaskSpec


class MetricSpecTests(unittest.TestCase):
    def test_from_dict_coerces_numeric_target(self) -> None:
        metric = MetricSpec.from_dict(
            {
                "name": "accuracy",
                "description": "Validation accuracy",
                "kind": "maximize",
                "target": "0.92",
            },
            priority=1,
        )

        self.assertEqual(metric.kind, "maximize")
        self.assertEqual(metric.target, 0.92)
        self.assertEqual(metric.priority, 1)

    def test_from_dict_coerces_pass_fail_target(self) -> None:
        metric = MetricSpec.from_dict(
            {
                "name": "tests_pass",
                "description": "All tests pass",
                "kind": "pass_fail",
                "target": "yes",
            },
            priority=2,
        )

        self.assertEqual(metric.target, True)
        self.assertEqual(metric.target_text(), "must pass")

    def test_from_dict_corrects_accuracy_direction(self) -> None:
        metric = MetricSpec.from_dict(
            {
                "name": "Validation Accuracy",
                "description": "Validation accuracy should be at least 0.95.",
                "kind": "minimize",
                "target": 0.95,
            },
            priority=1,
        )

        self.assertEqual(metric.kind, "maximize")

    def test_from_dict_corrects_runtime_direction(self) -> None:
        metric = MetricSpec.from_dict(
            {
                "name": "Execution Time",
                "description": "Total runtime should not exceed 2 seconds.",
                "kind": "maximize",
                "target": 2.0,
            },
            priority=2,
        )

        self.assertEqual(metric.kind, "minimize")


class TaskSpecTests(unittest.TestCase):
    def test_success_metrics_property_maps_metric_descriptions(self) -> None:
        spec = TaskSpec(
            original_prompt="sort a list",
            refined_description="Sort a list.",
            language="python",
            task_type="function",
            metrics=[
                MetricSpec(
                    name="sorted",
                    description="Returns sorted output",
                    kind="pass_fail",
                    target=True,
                    priority=1,
                )
            ],
        )

        self.assertEqual(spec.success_metrics, ["Returns sorted output"])
