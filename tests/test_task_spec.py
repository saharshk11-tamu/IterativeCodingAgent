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
