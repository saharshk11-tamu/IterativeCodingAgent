from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import subprocess
import shutil
import uuid

from agent.tools import WorkspaceRuntime
from agent.task_spec import MetricSpec


class WorkspaceRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path.cwd()
        self.workspace_rel = Path("workspace") / f"runtime-test-{uuid.uuid4().hex}"
        self.workspace_root = self.project_root / self.workspace_rel
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.runtime = WorkspaceRuntime(
            workspace_dir=self.workspace_rel,
            metric_specs=[
                MetricSpec("tests", "All tests pass", "pass_fail", True, 1),
            ],
            project_root=self.project_root,
            uv_executable="uv",
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_workspace_write_and_read_round_trip(self) -> None:
        self.runtime.write_file("solution.py", "print('hi')\n")
        outcome = self.runtime.read_file("solution.py")

        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.data["content"], "print('hi')\n")

    def test_workspace_path_enforcement_blocks_escape(self) -> None:
        with self.assertRaises(ValueError):
            self.runtime.write_file("../escape.py", "nope")

    def test_install_python_packages_rejects_invalid_spec(self) -> None:
        outcome = self.runtime.install_python_packages(["bad spec; rm -rf /"])

        self.assertFalse(outcome.ok)
        self.assertIn("Invalid package spec", outcome.summary)

    @patch("agent.tools.runtime.subprocess.run")
    def test_install_python_packages_uses_uv_and_records_state(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["uv", "run"],
            returncode=0,
            stdout="ok\n",
            stderr="",
        )

        outcome = self.runtime.install_python_packages(["numpy==1.26.4"])

        self.assertTrue(outcome.ok)
        self.assertIn("numpy==1.26.4", self.runtime.extra_packages)
        command = mock_run.call_args.kwargs["args"] if "args" in mock_run.call_args.kwargs else mock_run.call_args.args[0]
        self.assertEqual(command[:2], ["uv", "run"])
        self.assertIn("--with", command)

        history_path = self.workspace_root / ".agent" / "installed_packages.json"
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["active_specs"], ["numpy==1.26.4"])

    @patch.object(WorkspaceRuntime, "run_python_script")
    def test_run_evaluator_parses_json_output(self, mock_run_python_script) -> None:
        evaluator_path = self.workspace_root / ".agent" / "evaluator.py"
        evaluator_path.write_text("# stub\n", encoding="utf-8")
        mock_run_python_script.return_value.ok = True
        mock_run_python_script.return_value.summary = "ok"
        mock_run_python_script.return_value.data = {
            "stdout": json.dumps(
                {
                    "metrics": [
                        {
                            "name": "tests",
                            "kind": "pass_fail",
                            "value": True,
                            "target": True,
                            "met": True,
                        }
                    ],
                    "all_targets_met": True,
                    "summary": "done",
                }
            )
        }

        result = self.runtime.run_evaluator()

        self.assertTrue(result.all_targets_met)
        self.assertEqual(result.metrics[0].name, "tests")
