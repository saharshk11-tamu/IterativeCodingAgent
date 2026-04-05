from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.automation import AutomationOrchestrator
from agent.evaluation import EvaluationResult, MetricValue
from agent.task_spec import MetricSpec, TaskSpec
from agent.tools import ToolOutcome


class _FakeBridge:
    def __init__(self, responses: list[str], answers: list[str] | None = None) -> None:
        self._responses = list(responses)
        self._answers = list(answers or [])
        self.activities: list[tuple[str, str]] = []
        self.messages: list[str] = []
        self.questions: list[str] = []

    def ask_user(self, question: str) -> str:
        self.questions.append(question)
        if not self._answers:
            raise AssertionError(f"Unexpected question: {question}")
        return self._answers.pop(0)

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

    def workspace_entries(self):
        entries = set()
        for path in self.files:
            top_level = path.split("/", 1)[0]
            entries.add(top_level + "/" if "/" in path else top_level)
        for path in self.artifacts:
            top_level = path.split("/", 1)[0]
            entries.add(top_level + "/" if "/" in path else top_level)
        return sorted(entries)

    def clear_workspace(self):
        self.files = {}
        self.artifacts = {}
        self.extra_packages = []
        self.install_history = []

    def reset_agent_state(self):
        self.artifacts = {}
        self.extra_packages = []
        self.install_history = []

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
        if len(self._eval_results) > 1:
            return self._eval_results.pop(0)
        return self._eval_results[0]

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
        self.assertEqual(report["outcome"], "targets_met")
        self.assertEqual(report["workspace_action"], "empty")
        self.assertIn("Workspace: started from an empty workspace", report["final_summary"])
        self.assertIn("Outcome: met all confirmed criteria.", report["final_summary"])
        self.assertIn("tests: met", report["final_summary"])
        self.assertEqual(report["final_summary"], bridge.messages[-1])

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
        self.assertEqual(report["outcome"], "stopped_early")
        self.assertIn("Problem: required inputs or assets were missing from the workspace", report["final_summary"])
        self.assertIn("Latest issue: dataset missing", report["final_summary"])
        self.assertIn(("error", "dataset missing"), bridge.activities)

    def test_orchestrator_asks_about_existing_workspace_and_keeps_starter_code(self) -> None:
        bridge = _FakeBridge(
            [
                '{"status":"ready","summary":"workspace ready","relevant_paths":["solution.py"]}',
                '{"summary":"evaluator ready","files":[{"path":"tests/test_solution.py","content":"import unittest\\n"},{"path":".agent/evaluator.py","content":"print(1)\\n"}]}',
                '{"type":"tool_call","tool":"read_file","args":{"path":"solution.py"},"reason":"inspect starter code"}',
                '{"type":"tool_call","tool":"run_evaluator","args":{},"reason":"check metrics"}',
                '{"type":"finish","summary":"done"}',
            ],
            answers=["starter code"],
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

        runtime = _FakeRuntime("./workspace", spec.metrics)
        runtime.files = {"solution.py": "def solve():\n    return 0\n"}

        with patch("agent.automation.WorkspaceRuntime", return_value=runtime):
            report = AutomationOrchestrator(bridge).run(spec, workspace_dir="./workspace")

        self.assertEqual(report["workspace_action"], "starter_code")
        self.assertIn("solution.py", report["workspace_initial_entries"])
        self.assertTrue(bridge.questions)
        self.assertIn("workspace/ already contains files", bridge.questions[0])
        self.assertIn(("status", "keeping existing workspace files as starter code"), bridge.activities)

    def test_orchestrator_can_clear_existing_workspace(self) -> None:
        bridge = _FakeBridge(
            [
                '{"status":"ready","summary":"workspace ready","relevant_paths":[]}',
                '{"summary":"evaluator ready","files":[{"path":"tests/test_solution.py","content":"import unittest\\n"},{"path":".agent/evaluator.py","content":"print(1)\\n"}]}',
                '{"type":"tool_call","tool":"write_file","args":{"path":"solution.py","content":"def solve():\\n    return 1\\n"},"reason":"create implementation"}',
                '{"type":"tool_call","tool":"run_evaluator","args":{},"reason":"check metrics"}',
                '{"type":"finish","summary":"done"}',
            ],
            answers=["clear"],
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

        runtime = _FakeRuntime("./workspace", spec.metrics)
        runtime.files = {
            "old_solution.py": "raise NotImplementedError\n",
            "tests/old_test.py": "import unittest\n",
        }
        runtime.artifacts = {"run_report.json": {"old": True}}

        with patch("agent.automation.WorkspaceRuntime", return_value=runtime):
            report = AutomationOrchestrator(bridge).run(spec, workspace_dir="./workspace")

        self.assertEqual(report["workspace_action"], "cleared")
        self.assertNotIn("old_solution.py", runtime.files)
        self.assertIn(("status", "cleared workspace directory before starting the run"), bridge.activities)

    def test_orchestrator_retries_invalid_tool_json(self) -> None:
        bridge = _FakeBridge(
            [
                '{"status":"ready","summary":"workspace ready","relevant_paths":[]}',
                '{"summary":"evaluator ready","files":[{"path":"tests/test_solution.py","content":"import unittest\\n"},{"path":".agent/evaluator.py","content":"print(1)\\n"}]}',
                "this is not json",
                '{"type":"finish","summary":"stop for now"}',
                '{"type":"finish","summary":"stop for now"}',
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

        runtime = _FakeRuntime("./workspace", spec.metrics)
        runtime._eval_results = [
            EvaluationResult(
                metrics=[
                    MetricValue("tests", "pass_fail", False, True, False),
                ],
                all_targets_met=False,
                summary="baseline failed",
            ),
            EvaluationResult(
                metrics=[
                    MetricValue("tests", "pass_fail", False, True, False),
                ],
                all_targets_met=False,
                summary="still failing",
            ),
            EvaluationResult(
                metrics=[
                    MetricValue("tests", "pass_fail", False, True, False),
                ],
                all_targets_met=False,
                summary="still failing",
            ),
        ]

        with patch("agent.automation.WorkspaceRuntime", return_value=runtime):
            report = AutomationOrchestrator(bridge).run(spec, workspace_dir="./workspace")

        self.assertEqual(report["stop_reason"], "plateau")
        self.assertTrue(
            any("tool response was invalid" in text for kind, text in bridge.activities if kind == "error")
        )

    @patch("agent.automation.WorkspaceRuntime", _FakeRuntime)
    def test_orchestrator_regenerates_invalid_evaluator(self) -> None:
        bridge = _FakeBridge(
            [
                '{"status":"ready","summary":"workspace ready","relevant_paths":[]}',
                '{"summary":"broken evaluator","files":[{"path":"tests/test_solution.py","content":"import unittest\\n"},{"path":".agent/evaluator.py","content":"BROKEN"}]}',
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

        runtime = _FakeRuntime("./workspace", spec.metrics)
        runtime._eval_results = [
            EvaluationResult(
                metrics=[
                    MetricValue("tests", "pass_fail", False, True, False),
                ],
                all_targets_met=False,
                summary="Evaluator execution failed: broken",
            ),
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

        with patch("agent.automation.WorkspaceRuntime", return_value=runtime):
            report = AutomationOrchestrator(bridge).run(spec, workspace_dir="./workspace")

        self.assertEqual(report["stop_reason"], "all_targets_met")
        self.assertTrue(
            any("evaluator validation failed" in text for kind, text in bridge.activities if kind == "error")
        )
