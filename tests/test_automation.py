from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.automation import AutomationOrchestrator
from agent.evaluation import EvaluationResult, MetricValue
from agent.task_spec import MetricSpec, TaskSpec
from agent.tools import ToolOutcome


class _FakeBridge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.activities: list[tuple[str, str]] = []
        self.messages: list[str] = []

    def ask_user(self, question: str) -> str:
        raise AssertionError(f"Unexpected question: {question}")

    def post_activity(self, kind: str, text: str) -> None:
        self.activities.append((kind, text))

    def post_token(self, token: str) -> None:
        self.messages.append(token)

    def post_message(self, text: str) -> None:
        self.messages.append(text)

    def call_llm(self, messages: list[dict]) -> str:
        if not self._responses:
            raise AssertionError("No scripted LLM response left")
        return self._responses.pop(0)


class _FakeRuntime:
    def __init__(self, workspace_dir, metric_specs, **_kwargs) -> None:
        self.metric_specs = metric_specs
        self.extra_packages: list[str] = []
        self.install_history: list[dict[str, object]] = []
        self.files: dict[str, str] = {}
        self.artifacts: dict[str, object] = {}
        self._eval_results = [
            EvaluationResult(
                metrics=[
                    MetricValue("tests", "pass_fail", False, True, False),
                ],
                all_targets_met=False,
                summary="baseline failed",
            ),
            EvaluationResult(
                metrics=[
                    MetricValue("tests", "pass_fail", True, True, True),
                ],
                all_targets_met=True,
                summary="tests passed",
            ),
        ]

    def write_json_artifact(self, relative_path, payload) -> None:
        self.artifacts[relative_path] = payload

    def workspace_overview(self):
        return {
            "workspace_root": "workspace",
            "files": sorted(self.files),
            "extra_packages": list(self.extra_packages),
        }

    def read_file(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return ToolOutcome(True, f"Read {path}", {"content": self.files[path]})

    def write_file(self, path, content):
        self.files[path] = content
        return ToolOutcome(True, f"Wrote {path}", {"bytes": len(content)})

    def snapshot_touched_state(self):
        return dict(self.files)

    def restore_touched_state(self, snapshot):
        self.files = dict(snapshot)

    def run_evaluator(self):
        return self._eval_results.pop(0)

    def list_files(self, path="."):
        return ToolOutcome(True, "listed", {"entries": sorted(self.files)})

    def replace_in_file(self, path, search, replace):
        self.files[path] = self.files[path].replace(search, replace)
        return ToolOutcome(True, f"Updated {path}")

    def remove_path(self, path):
        self.files.pop(path, None)
        return ToolOutcome(True, f"Removed {path}")

    def install_python_packages(self, specs):
        self.extra_packages.extend(specs)
        self.install_history.append({"specs": specs, "ok": True, "summary": "installed"})
        return ToolOutcome(True, "installed", {"specs": specs})

    def run_python_tests(self, paths=None):
        return ToolOutcome(True, "tests ran", {"paths": paths or []})

    def run_python_script(self, path, args=None):
        return ToolOutcome(True, "script ran", {"path": path, "args": args or []})


class AutomationOrchestratorTests(unittest.TestCase):
    @patch("agent.automation.WorkspaceRuntime", _FakeRuntime)
    def test_orchestrator_runs_to_metric_target(self) -> None:
        bridge = _FakeBridge(
            [
                '{"status":"ready","summary":"workspace ready","relevant_paths":[]}',
                '{"summary":"evaluator ready","files":[{"path":"tests/test_solution.py","content":"import unittest\\n"},{"path":".agent/evaluator.py","content":"print(1)\\n"}]}',
                '{"type":"tool_call","tool":"write_file","args":{"path":"solution.py","content":"def solve():\\n    return 1\\n"},"reason":"create implementation"}',
                '{"type":"tool_call","tool":"run_evaluator","args":{},"reason":"check metrics"}',
                '{"type":"finish","summary":"done"}',
            ]
        )
        spec = TaskSpec(
            original_prompt="write a function",
            refined_description="Write a function.",
            language="python",
            task_type="function",
            metrics=[
                MetricSpec("tests", "All tests pass", "pass_fail", True, 1),
            ],
        )

        report = AutomationOrchestrator(bridge).run(spec, workspace_dir="./workspace")

        self.assertEqual(report["stop_reason"], "all_targets_met")
        self.assertTrue(report["best_result"]["all_targets_met"])
        self.assertIn("Automation run finished.", bridge.messages[-1])

    @patch("agent.automation.WorkspaceRuntime", _FakeRuntime)
    def test_orchestrator_stops_when_grounding_blocks(self) -> None:
        bridge = _FakeBridge(
            [
                '{"status":"blocked","summary":"dataset missing","relevant_paths":[]}',
            ]
        )
        spec = TaskSpec(
            original_prompt="train a model on images",
            refined_description="Train a model.",
            language="python",
            task_type="data_pipeline",
            metrics=[
                MetricSpec("accuracy", "Validation accuracy", "maximize", 0.9, 1),
            ],
        )

        report = AutomationOrchestrator(bridge).run(spec, workspace_dir="./workspace")

        self.assertEqual(report["stop_reason"], "blocked_inputs")
        self.assertIn(("error", "dataset missing"), bridge.activities)
