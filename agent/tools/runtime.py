"""Workspace-only tool runtime for the automation loop."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.evaluation import EvaluationResult, make_failure_result, parse_evaluation_result
from agent.task_spec import MetricSpec

MAX_TOOL_SECONDS = 600
_VALID_PACKAGE_SPEC = re.compile(r"^[A-Za-z0-9_.-]+([<>=!~]=?[A-Za-z0-9*+_.-]+)?$")


@dataclass
class ToolOutcome:
    ok: bool
    summary: str
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "data": self.data or {},
        }


class WorkspaceRuntime:
    def __init__(
        self,
        workspace_dir: str | Path,
        metric_specs: list[MetricSpec],
        *,
        project_root: str | Path | None = None,
        uv_executable: str | None = None,
    ) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.workspace_root = (self.project_root / workspace_dir).resolve()
        self.agent_root = self.workspace_root / ".agent"
        self.metric_specs = metric_specs
        self.uv_executable = uv_executable or shutil.which("uv") or "uv"
        self.extra_packages: list[str] = []
        self.install_history: list[dict[str, Any]] = []
        self._touched_paths: set[str] = set()

        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.agent_root.mkdir(parents=True, exist_ok=True)
        self._load_install_history()

    def workspace_entries(self) -> list[str]:
        entries: list[str] = []
        for child in sorted(self.workspace_root.iterdir(), key=lambda item: item.name):
            suffix = "/" if child.is_dir() else ""
            entries.append(child.name + suffix)
        return entries

    def clear_workspace(self) -> None:
        for child in list(self.workspace_root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        self._reset_runtime_state()
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.agent_root.mkdir(parents=True, exist_ok=True)

    def reset_agent_state(self) -> None:
        if self.agent_root.exists():
            shutil.rmtree(self.agent_root)
        self._reset_runtime_state()
        self.agent_root.mkdir(parents=True, exist_ok=True)

    def list_files(self, path: str = ".") -> ToolOutcome:
        root = self._resolve_path(path, allow_missing=False)
        if root.is_file():
            rel = self._to_workspace_path(root)
            return ToolOutcome(True, f"Listed file {rel}", {"entries": [rel]})

        entries: list[str] = []
        for current in sorted(root.rglob("*")):
            rel = self._to_workspace_path(current)
            if rel.startswith(".agent/"):
                continue
            suffix = "/" if current.is_dir() else ""
            entries.append(rel + suffix)
        return ToolOutcome(True, f"Listed {len(entries)} path(s)", {"entries": entries})

    def read_file(self, path: str) -> ToolOutcome:
        resolved = self._resolve_path(path, allow_missing=False)
        if resolved.is_dir():
            return ToolOutcome(False, f"{path} is a directory")
        return ToolOutcome(
            True,
            f"Read {self._to_workspace_path(resolved)}",
            {"content": resolved.read_text(encoding="utf-8")},
        )

    def write_file(self, path: str, content: str) -> ToolOutcome:
        resolved = self._resolve_path(path, allow_missing=True)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        self._mark_touched(resolved)
        return ToolOutcome(
            True,
            f"Wrote {self._to_workspace_path(resolved)}",
            {"bytes": len(content.encode("utf-8"))},
        )

    def replace_in_file(self, path: str, search: str, replace: str) -> ToolOutcome:
        resolved = self._resolve_path(path, allow_missing=False)
        content = resolved.read_text(encoding="utf-8")
        if search not in content:
            return ToolOutcome(False, f"Search text not found in {path}")
        updated = content.replace(search, replace)
        resolved.write_text(updated, encoding="utf-8")
        self._mark_touched(resolved)
        return ToolOutcome(
            True,
            f"Updated {self._to_workspace_path(resolved)}",
            {"replacements": content.count(search)},
        )

    def remove_path(self, path: str) -> ToolOutcome:
        resolved = self._resolve_path(path, allow_missing=False)
        rel = self._to_workspace_path(resolved)
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        self._touched_paths.add(rel)
        return ToolOutcome(True, f"Removed {rel}")

    def install_python_packages(self, specs: list[str]) -> ToolOutcome:
        if not specs:
            return ToolOutcome(False, "No package specs provided")

        cleaned_specs = []
        for spec in specs:
            candidate = spec.strip()
            if not candidate or not _VALID_PACKAGE_SPEC.match(candidate):
                return ToolOutcome(False, f"Invalid package spec: {spec}")
            cleaned_specs.append(candidate)

        command = self._uv_python_command(["-c", "print('ok')"], with_specs=cleaned_specs)
        outcome = self._run_subprocess(command, timeout=120)

        history_record = {
            "specs": cleaned_specs,
            "ok": outcome.ok,
            "summary": outcome.summary,
        }
        self.install_history.append(history_record)
        if outcome.ok:
            for spec in cleaned_specs:
                if spec not in self.extra_packages:
                    self.extra_packages.append(spec)
        self._write_installed_packages()
        return ToolOutcome(outcome.ok, outcome.summary, {"specs": cleaned_specs})

    def run_python_tests(self, paths: list[str] | None = None) -> ToolOutcome:
        if paths:
            relative_paths = [
                self._to_workspace_path(self._resolve_path(path, allow_missing=False))
                for path in paths
            ]
            command = self._uv_python_command(["-m", "unittest", "-v", *relative_paths])
        else:
            default_dir = self.workspace_root / "tests"
            if not default_dir.exists():
                return ToolOutcome(False, "workspace/tests does not exist")
            command = self._uv_python_command(
                ["-m", "unittest", "discover", "-s", str(default_dir)]
            )
        return self._run_subprocess(command, timeout=MAX_TOOL_SECONDS)

    def run_python_script(self, path: str, args: list[str] | None = None) -> ToolOutcome:
        resolved = self._resolve_path(path, allow_missing=False)
        command = self._uv_python_command([str(resolved), *(args or [])])
        return self._run_subprocess(command, timeout=MAX_TOOL_SECONDS)

    def run_evaluator(self) -> EvaluationResult:
        evaluator_path = self.agent_root / "evaluator.py"
        if not evaluator_path.exists():
            return make_failure_result(self.metric_specs, "Evaluator file is missing.")

        outcome = self.run_python_script(".agent/evaluator.py")
        if not outcome.ok:
            return make_failure_result(
                self.metric_specs,
                f"Evaluator execution failed: {outcome.summary}",
            )

        stdout = str((outcome.data or {}).get("stdout", "")).strip()
        try:
            return parse_evaluation_result(stdout)
        except Exception as exc:  # noqa: BLE001
            return make_failure_result(
                self.metric_specs,
                f"Evaluator returned invalid JSON: {exc}",
            )

    def snapshot_touched_state(self) -> dict[str, str | None]:
        snapshot: dict[str, str | None] = {}
        for rel_path in sorted(self._touched_paths):
            path = self.workspace_root / rel_path
            if path.exists():
                snapshot[rel_path] = path.read_text(encoding="utf-8")
            else:
                snapshot[rel_path] = None
        return snapshot

    def restore_touched_state(self, snapshot: dict[str, str | None]) -> None:
        for rel_path, content in snapshot.items():
            resolved = self._resolve_path(rel_path, allow_missing=True)
            if content is None:
                if resolved.exists():
                    if resolved.is_dir():
                        shutil.rmtree(resolved)
                    else:
                        resolved.unlink()
                continue
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")

    def write_json_artifact(
        self,
        relative_path: str,
        payload: dict[str, Any] | list[Any],
    ) -> None:
        resolved = self._resolve_agent_artifact(relative_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def workspace_overview(self) -> dict[str, Any]:
        files = self.list_files(".").to_dict().get("data", {}).get("entries", [])
        return {
            "workspace_root": str(self.workspace_root),
            "files": files,
            "extra_packages": list(self.extra_packages),
        }

    def _resolve_path(self, path: str, *, allow_missing: bool) -> Path:
        candidate = (self.workspace_root / path).resolve()
        if not str(candidate).startswith(str(self.workspace_root)):
            raise ValueError(f"path escapes workspace: {path}")
        if not allow_missing and not candidate.exists():
            raise FileNotFoundError(path)
        return candidate

    def _resolve_agent_artifact(self, path: str) -> Path:
        candidate = (self.agent_root / path).resolve()
        if not str(candidate).startswith(str(self.agent_root)):
            raise ValueError(f"artifact path escapes .agent: {path}")
        return candidate

    def _to_workspace_path(self, path: Path) -> str:
        return path.relative_to(self.workspace_root).as_posix()

    def _mark_touched(self, path: Path) -> None:
        self._touched_paths.add(self._to_workspace_path(path))

    def _load_install_history(self) -> None:
        history_path = self.agent_root / "installed_packages.json"
        if not history_path.exists():
            return
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        history = data.get("history", [])
        if isinstance(history, list):
            self.install_history = [item for item in history if isinstance(item, dict)]
        specs = data.get("active_specs", [])
        if isinstance(specs, list):
            self.extra_packages = [str(item) for item in specs]

    def _write_installed_packages(self) -> None:
        self.write_json_artifact(
            "installed_packages.json",
            {
                "active_specs": self.extra_packages,
                "history": self.install_history,
            },
        )

    def _uv_python_command(
        self,
        python_args: list[str],
        *,
        with_specs: list[str] | None = None,
    ) -> list[str]:
        command = [self.uv_executable, "run"]
        for spec in [*(self.extra_packages), *((with_specs or []))]:
            command.extend(["--with", spec])
        command.extend(["python", *python_args])
        return command

    def _run_subprocess(self, command: list[str], *, timeout: int) -> ToolOutcome:
        try:
            completed = subprocess.run(
                command,
                cwd=self.project_root,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return ToolOutcome(False, f"Required executable not found: {command[0]}")
        except subprocess.TimeoutExpired:
            return ToolOutcome(False, f"Command timed out after {timeout} seconds")

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        summary = f"exit code {completed.returncode}"
        if stderr:
            summary = f"{summary}; stderr: {stderr}"
        elif stdout:
            summary = f"{summary}; stdout: {stdout}"
        return ToolOutcome(
            completed.returncode == 0,
            summary,
            {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": completed.returncode,
                "command": command,
            },
        )

    def _reset_runtime_state(self) -> None:
        self.extra_packages = []
        self.install_history = []
        self._touched_paths.clear()
