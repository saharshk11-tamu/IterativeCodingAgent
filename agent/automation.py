"""Autonomous workspace workflow built on fixed runtime tools."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from agent.evaluation import (
    EvaluationResult,
    compare_results,
    format_evaluation_table,
)
from agent.intake import BridgeProtocol
from agent.task_spec import TaskSpec
from agent.tools import ToolOutcome, WorkspaceRuntime, execute_tool

_GROUNDING_SYSTEM = """\
You are grounding a coding task against the current workspace.

Given:
- a structured task specification
- the current file tree in workspace/

Output ONLY a JSON object with:
{
  "status": "ready" | "blocked",
  "summary": "<one short sentence>",
  "relevant_paths": ["<workspace relative path>", ...]
}

Rules:
- Return "blocked" only if the task clearly requires external files or assets
  that are missing from the workspace.
- A fresh coding task with no starter code is still "ready".
- Keep relevant_paths short and focused. Include only files that should be read
  before evaluator generation or implementation begins."""

_EVALUATOR_SYSTEM = """\
You are creating executable evaluation artifacts for a Python coding task.

Return ONLY a JSON object with:
{
  "summary": "<one short sentence>",
  "files": [
    {"path": "tests/test_solution.py", "content": "<file contents>"},
    {"path": ".agent/evaluator.py", "content": "<file contents>"}
  ]
}

Rules:
- Python only.
- Use standard library testing where practical.
- The evaluator must print exactly one JSON object to stdout with keys:
  metrics, all_targets_met, summary
- The evaluator must never crash the run on missing implementation files,
  import errors, or test failures. It should catch those cases and emit metric
  failures in JSON instead.
- Default implementation target is workspace/solution.py unless existing
  starter files make a different target obvious.
- The evaluator must add the workspace root to sys.path before imports.
- Do not write conversational text outside the JSON object."""

_AUTOMATION_SYSTEM = """\
You are the autonomous implementation phase of a coding agent.

You may use ONLY the provided tool API. You do NOT have shell access.
All edits must stay inside workspace/.
Do NOT modify files under .agent/ or tests/ during the implementation loop.

Respond with EXACTLY one JSON object in one of these forms:

Tool call:
{
  "type": "tool_call",
  "tool": "<tool name>",
  "args": { ... },
  "reason": "<short reason>"
}

Finish the iteration:
{
  "type": "finish",
  "summary": "<short summary of the best next stopping point>"
}

Available tools:
- list_files(path=".")
- read_file(path)
- write_file(path, content)
- replace_in_file(path, search, replace)
- remove_path(path)
- install_python_packages(specs)
- run_evaluator()
- run_python_tests(paths=None)
- run_python_script(path, args=None)

Rules:
- Work Python-first.
- Prefer inspecting files before editing them.
- Ensure every implementation iteration ends with an evaluator run.
- If a dependency is required, call install_python_packages with exact package
  specifiers. Do not mention shell commands.
- Keep edits focused on satisfying the confirmed metrics.
- Output JSON only."""


@dataclass(frozen=True)
class GroundingResult:
    status: str
    summary: str
    relevant_paths: list[str]


class AutomationOrchestrator:
    MAX_ITERATIONS = 8
    PLATEAU_LIMIT = 2
    MAX_TOOL_ACTIONS_PER_ITERATION = 12
    MAX_RUN_SECONDS = 45 * 60

    def __init__(self, bridge: BridgeProtocol) -> None:
        self._bridge = bridge

    def run(self, spec: TaskSpec, workspace_dir: str = "./workspace") -> dict[str, Any]:
        runtime = WorkspaceRuntime(workspace_dir=workspace_dir, metric_specs=spec.metrics)
        run_started = time.monotonic()

        runtime.write_json_artifact("task_spec.json", spec.to_dict())
        runtime.write_json_artifact(
            "metrics.json",
            [metric.to_dict() for metric in spec.metrics],
        )

        if spec.language.lower() != "python":
            summary = f"Automation currently supports Python only; received {spec.language}."
            self._bridge.post_activity("error", summary)
            report = self._write_blocked_report(runtime, spec, summary, "unsupported_language")
            self._bridge.post_message(self._final_summary(report))
            return report

        grounding = self._ground_workspace(spec, runtime)
        if grounding.status == "blocked":
            self._bridge.post_activity("error", grounding.summary)
            report = self._write_blocked_report(runtime, spec, grounding.summary, "blocked_inputs")
            self._bridge.post_message(self._final_summary(report))
            return report

        self._bridge.post_activity("status", "generating evaluator artifacts...")
        evaluator_files = self._generate_evaluator(spec, runtime, grounding)
        for file_spec in evaluator_files["files"]:
            runtime.write_file(file_spec["path"], file_spec["content"])
        self._bridge.post_activity(
            "tool_result",
            f"evaluator ready: {', '.join(file['path'] for file in evaluator_files['files'])}",
        )

        baseline = runtime.run_evaluator()
        self._bridge.post_activity("metrics", f"baseline\n{format_evaluation_table(baseline)}")

        best_result = baseline
        best_snapshot = runtime.snapshot_touched_state()
        iterations: list[dict[str, Any]] = []
        plateau_count = 0
        stop_reason = "all_targets_met" if baseline.all_targets_met else "iteration_budget"

        if baseline.all_targets_met:
            report = self._build_report(
                spec=spec,
                baseline=baseline,
                best_result=best_result,
                iterations=iterations,
                stop_reason=stop_reason,
                runtime=runtime,
            )
            runtime.write_json_artifact("run_report.json", report)
            self._bridge.post_message(self._final_summary(report))
            return report

        for iteration_index in range(1, self.MAX_ITERATIONS + 1):
            if time.monotonic() - run_started > self.MAX_RUN_SECONDS:
                stop_reason = "time_limit"
                break

            self._bridge.post_activity("status", f"starting iteration {iteration_index}...")
            iteration_data = self._run_iteration(
                spec=spec,
                runtime=runtime,
                grounding=grounding,
                best_result=best_result,
                baseline=baseline,
                iteration_index=iteration_index,
            )
            iterations.append(iteration_data)
            current_result = iteration_data["evaluation"]
            self._bridge.post_activity(
                "metrics",
                f"iteration {iteration_index}\n{format_evaluation_table(current_result)}",
            )

            comparison = compare_results(current_result, best_result, spec.metrics)
            if comparison > 0:
                best_result = current_result
                best_snapshot = runtime.snapshot_touched_state()
                plateau_count = 0
                iteration_data["improved"] = True
            else:
                plateau_count += 1
                runtime.restore_touched_state(best_snapshot)
                iteration_data["improved"] = False
            iteration_data["best_so_far"] = best_result

            if best_result.all_targets_met:
                stop_reason = "all_targets_met"
                break
            if plateau_count >= self.PLATEAU_LIMIT:
                stop_reason = "plateau"
                break
        else:
            stop_reason = "iteration_budget"

        runtime.restore_touched_state(best_snapshot)
        report = self._build_report(
            spec=spec,
            baseline=baseline,
            best_result=best_result,
            iterations=iterations,
            stop_reason=stop_reason,
            runtime=runtime,
        )
        runtime.write_json_artifact("run_report.json", report)
        self._bridge.post_activity("status", f"automation stopped: {stop_reason}")
        self._bridge.post_message(self._final_summary(report))
        return report

    def _ground_workspace(self, spec: TaskSpec, runtime: WorkspaceRuntime) -> GroundingResult:
        self._bridge.post_activity("status", "grounding task against workspace...")
        overview = runtime.workspace_overview()
        messages = [
            {"role": "system", "content": _GROUNDING_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_spec": spec.to_dict(),
                        "workspace": overview,
                    },
                    indent=2,
                ),
            },
        ]
        try:
            raw = self._bridge.call_llm(messages)
            data = self._parse_json_response(raw)
        except Exception as exc:  # noqa: BLE001
            self._bridge.post_activity(
                "error",
                f"grounding parse failed ({exc}); continuing with default workspace scan",
            )
            return GroundingResult(status="ready", summary="workspace ready", relevant_paths=[])

        return GroundingResult(
            status=str(data.get("status", "ready")).strip().lower(),
            summary=str(data.get("summary", "workspace ready")).strip() or "workspace ready",
            relevant_paths=[
                str(path)
                for path in data.get("relevant_paths", [])
                if isinstance(path, str)
            ],
        )

    def _generate_evaluator(
        self,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
    ) -> dict[str, Any]:
        context_files = []
        for path in grounding.relevant_paths[:6]:
            try:
                result = runtime.read_file(path)
            except Exception:  # noqa: BLE001
                continue
            if result.ok:
                context_files.append({"path": path, "content": result.data.get("content", "")})

        messages = [
            {"role": "system", "content": _EVALUATOR_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_spec": spec.to_dict(),
                        "workspace_files": runtime.workspace_overview()["files"],
                        "context_files": context_files,
                    },
                    indent=2,
                ),
            },
        ]

        raw = self._bridge.call_llm(messages)
        data = self._parse_json_response(raw)
        files = data.get("files")
        if not isinstance(files, list):
            raise ValueError("evaluator response did not include files")
        normalized_files: list[dict[str, str]] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))
            if not path or not content:
                continue
            normalized_files.append({"path": path, "content": content})
        if not normalized_files:
            raise ValueError("evaluator response included no writable files")
        return {
            "summary": str(data.get("summary", "")).strip(),
            "files": normalized_files,
        }

    def _run_iteration(
        self,
        *,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
        best_result: EvaluationResult,
        baseline: EvaluationResult,
        iteration_index: int,
    ) -> dict[str, Any]:
        history: list[dict[str, str]] = []
        evaluation_result: EvaluationResult | None = None
        changed_files: list[str] = []
        package_specs: list[str] = []

        for action_index in range(1, self.MAX_TOOL_ACTIONS_PER_ITERATION + 1):
            tool_request = self._next_tool_request(
                spec=spec,
                runtime=runtime,
                grounding=grounding,
                baseline=baseline,
                best_result=best_result,
                iteration_index=iteration_index,
                action_index=action_index,
                history=history,
            )

            if tool_request.get("type") == "finish":
                history.append({"role": "assistant", "content": json.dumps(tool_request)})
                break

            tool_name = str(tool_request.get("tool", "")).strip()
            args = tool_request.get("args", {})
            reason = str(tool_request.get("reason", "")).strip()
            history.append({"role": "assistant", "content": json.dumps(tool_request)})
            self._bridge.post_activity("tool_call", f"{tool_name} - {reason or 'requested'}")

            outcome, maybe_result = execute_tool(tool_name, args, runtime)
            summary_text = self._tool_result_for_model(outcome, maybe_result)
            history.append({"role": "user", "content": summary_text})
            self._bridge.post_activity("tool_result", outcome.summary)

            if outcome.ok and tool_name in {"write_file", "replace_in_file", "remove_path"}:
                path = str(args.get("path", "")).strip()
                if path and path not in changed_files:
                    changed_files.append(path)
            if outcome.ok and tool_name == "install_python_packages":
                specs = args.get("specs", [])
                if isinstance(specs, list):
                    for spec in specs:
                        spec_text = str(spec)
                        if spec_text not in package_specs:
                            package_specs.append(spec_text)
            if maybe_result is not None:
                evaluation_result = maybe_result

        if evaluation_result is None:
            self._bridge.post_activity("tool_call", "run_evaluator - forced end-of-iteration evaluation")
            evaluation_result = runtime.run_evaluator()
            self._bridge.post_activity(
                "tool_result",
                f"forced evaluation complete - {evaluation_result.summary or 'metrics collected'}",
            )

        return {
            "iteration": iteration_index,
            "evaluation": evaluation_result,
            "summary": evaluation_result.summary,
            "changed_files": changed_files,
            "package_specs": package_specs,
        }

    def _next_tool_request(
        self,
        *,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
        baseline: EvaluationResult,
        best_result: EvaluationResult,
        iteration_index: int,
        action_index: int,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _AUTOMATION_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "iteration": iteration_index,
                        "action_index": action_index,
                        "task_spec": spec.to_dict(),
                        "workspace_overview": runtime.workspace_overview(),
                        "grounding_summary": grounding.summary,
                        "baseline": baseline.to_dict(),
                        "best_result": best_result.to_dict(),
                    },
                    indent=2,
                ),
            },
        ]
        messages.extend(history)

        raw = self._bridge.call_llm(messages)
        request = self._parse_json_response(raw)

        if request.get("type") == "tool_call":
            tool_name = str(request.get("tool", "")).strip()
            args = request.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("tool args must be an object")
            if tool_name in {"write_file", "replace_in_file", "remove_path"}:
                path = str(args.get("path", "")).strip()
                if path.startswith(".agent/") or path.startswith("tests/"):
                    return {
                        "type": "finish",
                        "summary": "Protected evaluation files should not be modified.",
                    }
        return request

    def _tool_result_for_model(
        self,
        outcome: ToolOutcome,
        evaluation: EvaluationResult | None,
    ) -> str:
        payload = outcome.to_dict()
        if evaluation is not None:
            payload["evaluation"] = evaluation.to_dict()
        text = json.dumps(payload, indent=2)
        return f"TOOL RESULT\n{text[:12000]}"

    def _write_blocked_report(
        self,
        runtime: WorkspaceRuntime,
        spec: TaskSpec,
        summary: str,
        stop_reason: str,
    ) -> dict[str, Any]:
        baseline = {
            "metrics": [],
            "all_targets_met": False,
            "summary": summary,
        }
        report = {
            "task_spec": spec.to_dict(),
            "baseline": baseline,
            "best_result": baseline,
            "iterations": [],
            "stop_reason": stop_reason,
            "installed_packages": list(runtime.extra_packages),
            "install_history": list(runtime.install_history),
        }
        runtime.write_json_artifact("run_report.json", report)
        return report

    def _build_report(
        self,
        *,
        spec: TaskSpec,
        baseline: EvaluationResult,
        best_result: EvaluationResult,
        iterations: list[dict[str, Any]],
        stop_reason: str,
        runtime: WorkspaceRuntime,
    ) -> dict[str, Any]:
        serialized_iterations = []
        for item in iterations:
            serialized_iterations.append(
                {
                    "iteration": item["iteration"],
                    "summary": item["summary"],
                    "changed_files": item.get("changed_files", []),
                    "package_specs": item.get("package_specs", []),
                    "improved": bool(item.get("improved", False)),
                    "evaluation": item["evaluation"].to_dict(),
                    "best_so_far": item["best_so_far"].to_dict(),
                }
            )

        return {
            "task_spec": spec.to_dict(),
            "baseline": baseline.to_dict(),
            "best_result": best_result.to_dict(),
            "iterations": serialized_iterations,
            "stop_reason": stop_reason,
            "installed_packages": list(runtime.extra_packages),
            "install_history": list(runtime.install_history),
        }

    def _final_summary(self, report: dict[str, Any]) -> str:
        best = report.get("best_result", {})
        metrics = best.get("metrics", [])
        lines = [
            "Automation run finished.",
            f"Stop reason: {report.get('stop_reason', 'unknown')}",
        ]
        if metrics:
            lines.append("Best metrics:")
            for metric in metrics:
                lines.append(
                    f"- {metric.get('name')}: value={metric.get('value')} target={metric.get('target')} met={metric.get('met')}"
                )
        installed = report.get("installed_packages", [])
        if installed:
            lines.append(f"Installed package specs: {', '.join(installed)}")
        history = report.get("install_history", [])
        if history:
            lines.append(f"Package install attempts: {len(history)}")
        lines.append("Artifacts written under workspace/.agent/")
        return "\n".join(lines)

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
