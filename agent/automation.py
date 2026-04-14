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

_REQUIRED_EVALUATOR_FILE_PATHS = ("tests/test_solution.py", ".agent/evaluator.py")

_EVALUATOR_PLAN_SYSTEM = """\
You are planning executable evaluation artifacts for a Python coding task.

Return ONLY a JSON object with:
{
  "summary": "<one short sentence>",
  "files": [
    {"path": "tests/test_solution.py", "purpose": "<short purpose>"},
    {"path": ".agent/evaluator.py", "purpose": "<short purpose>"}
  ]
}

Rules:
- JSON only.
- Include exactly these two file paths and no others:
  tests/test_solution.py
  .agent/evaluator.py
- Do not include file contents or code.
- Keep purpose text short and concrete.
- Default implementation target is workspace/solution.py unless existing
  starter files make a different target obvious."""

_EVALUATOR_TEST_FILE_SYSTEM = """\
You are writing the Python unit-test file for a coding-task evaluator.

Return ONLY the raw contents of tests/test_solution.py.
Do not include markdown fences, JSON, or commentary.

Rules:
- Python only.
- Use standard library unittest where practical.
- Add the workspace root to sys.path before importing the implementation.
- Use workspace-relative paths derived from __file__ or pathlib. Never hard-code
  absolute paths like /workspace/...
- Do not use typing generics such as List[int] inside isinstance checks.
- Default implementation target is workspace/solution.py unless existing
  starter files make a different target obvious."""

_EVALUATOR_RUNNER_FILE_SYSTEM = """\
You are writing the evaluator entrypoint for a Python coding task.

Return ONLY the raw contents of .agent/evaluator.py.
Do not include markdown fences, JSON, or commentary.

Rules:
- Python only.
- The evaluator must print exactly one JSON object to stdout with keys:
  metrics, all_targets_met, summary
- The metrics field must be a JSON array. Each item must have:
  name, kind, value, target, met
- The evaluator must never crash the run on missing implementation files,
  import errors, or test failures. It should catch those cases and emit metric
  failures in JSON instead.
- Use workspace-relative paths derived from __file__ or pathlib. Never hard-code
  absolute paths like /workspace/...
- Do not use typing generics such as List[int] inside isinstance checks.
- Default implementation target is workspace/solution.py unless existing
  starter files make a different target obvious."""

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


@dataclass(frozen=True)
class EvaluatorFilePlan:
    path: str
    purpose: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "purpose": self.purpose}


@dataclass(frozen=True)
class EvaluatorPlan:
    summary: str
    files: list[EvaluatorFilePlan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "files": [file.to_dict() for file in self.files],
        }


class AutomationOrchestrator:
    MAX_ITERATIONS = 8
    PLATEAU_LIMIT = 2
    MAX_TOOL_ACTIONS_PER_ITERATION = 12
    MAX_RUN_SECONDS = 45 * 60
    MAX_EVALUATOR_GENERATION_ATTEMPTS = 3
    MAX_EVALUATOR_FILE_GENERATION_ATTEMPTS = 2
    MAX_TOOL_REQUEST_RETRIES = 2

    def __init__(self, bridge: BridgeProtocol) -> None:
        self._bridge = bridge

    def run(self, spec: TaskSpec, workspace_dir: str = "./workspace") -> dict[str, Any]:
        runtime = WorkspaceRuntime(workspace_dir=workspace_dir, metric_specs=spec.metrics)
        run_started = time.monotonic()
        workspace_action, workspace_initial_entries = self._resolve_workspace_state(runtime)
        runtime.reset_agent_state()

        if spec.language.lower() != "python":
            summary = f"Automation currently supports Python only; received {spec.language}."
            self._bridge.post_activity("error", summary)
            report = self._finalize_report(
                self._write_blocked_report(runtime, spec, summary, "unsupported_language")
            )
            self._attach_workspace_state(report, workspace_action, workspace_initial_entries)
            runtime.write_json_artifact("run_report.json", report)
            self._bridge.post_message(report["final_summary"])
            return report

        runtime.write_json_artifact("task_spec.json", spec.to_dict())
        runtime.write_json_artifact(
            "metrics.json",
            [metric.to_dict() for metric in spec.metrics],
        )

        grounding = self._ground_workspace(spec, runtime)
        if grounding.status == "blocked":
            self._bridge.post_activity("error", grounding.summary)
            report = self._finalize_report(
                self._write_blocked_report(runtime, spec, grounding.summary, "blocked_inputs")
            )
            self._attach_workspace_state(report, workspace_action, workspace_initial_entries)
            runtime.write_json_artifact("run_report.json", report)
            self._bridge.post_message(report["final_summary"])
            return report

        self._bridge.post_activity("status", "generating evaluator artifacts...")
        try:
            evaluator_bundle = self._prepare_evaluator(spec, runtime, grounding)
        except ValueError as exc:
            summary = f"evaluator generation failed: {exc}"
            self._bridge.post_activity("error", summary)
            report = self._finalize_report(
                self._write_blocked_report(
                    runtime,
                    spec,
                    summary,
                    "evaluator_generation_failed",
                )
            )
            self._attach_workspace_state(report, workspace_action, workspace_initial_entries)
            runtime.write_json_artifact("run_report.json", report)
            self._bridge.post_message(report["final_summary"])
            return report

        baseline = evaluator_bundle["baseline"]
        self._bridge.post_activity("metrics", f"baseline\n{format_evaluation_table(baseline)}")

        best_result = baseline
        best_snapshot = runtime.snapshot_touched_state()
        iterations: list[dict[str, Any]] = []
        plateau_count = 0
        stop_reason = "all_targets_met" if baseline.all_targets_met else "iteration_budget"

        if baseline.all_targets_met:
            report = self._finalize_report(
                self._build_report(
                    spec=spec,
                    baseline=baseline,
                    best_result=best_result,
                    iterations=iterations,
                    stop_reason=stop_reason,
                    runtime=runtime,
                )
            )
            self._attach_workspace_state(report, workspace_action, workspace_initial_entries)
            runtime.write_json_artifact("run_report.json", report)
            self._bridge.post_message(report["final_summary"])
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
        report = self._finalize_report(
            self._build_report(
                spec=spec,
                baseline=baseline,
                best_result=best_result,
                iterations=iterations,
                stop_reason=stop_reason,
                runtime=runtime,
            )
        )
        self._attach_workspace_state(report, workspace_action, workspace_initial_entries)
        runtime.write_json_artifact("run_report.json", report)
        self._bridge.post_activity("status", f"automation stopped: {stop_reason}")
        self._bridge.post_message(report["final_summary"])
        return report

    def _resolve_workspace_state(
        self,
        runtime: WorkspaceRuntime,
    ) -> tuple[str, list[str]]:
        entries = runtime.workspace_entries()
        if not entries:
            return "empty", []

        preview = ", ".join(f"`{entry}`" for entry in entries[:8])
        if len(entries) > 8:
            preview = f"{preview}, ... (+{len(entries) - 8} more)"

        while True:
            answer = self._bridge.ask_user(
                "workspace/ already contains files: "
                f"{preview}\n\n"
                "Should I treat these as starter code or clear the directory before this run? "
                "Reply with 'starter code' or 'clear'."
            ).strip().lower()

            if any(keyword in answer for keyword in {"clear", "delete", "remove", "empty"}):
                runtime.clear_workspace()
                self._bridge.post_activity(
                    "status",
                    "cleared workspace directory before starting the run",
                )
                return "cleared", entries

            if (
                "starter" in answer
                or "keep" in answer
                or "use them" in answer
                or "use it" in answer
            ):
                self._bridge.post_activity(
                    "status",
                    "keeping existing workspace files as starter code",
                )
                return "starter_code", entries

            self._bridge.post_activity(
                "status",
                "workspace choice was unclear; please reply with 'starter code' or 'clear'",
            )

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

    def _evaluator_context_payload(
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

        return {
            "task_spec": spec.to_dict(),
            "workspace_files": runtime.workspace_overview()["files"],
            "context_files": context_files,
        }

    def _generate_evaluator_plan(
        self,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
        *,
        validation_feedback: str | None = None,
    ) -> EvaluatorPlan:
        payload = self._evaluator_context_payload(spec, runtime, grounding)
        if validation_feedback:
            payload["previous_generation_issue"] = validation_feedback

        messages = [
            {"role": "system", "content": _EVALUATOR_PLAN_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            },
        ]
        data = self._request_json_object(
            messages,
            correction_prompt=(
                "Your previous response was invalid. Reply with exactly one JSON object "
                "matching the evaluator plan schema. Include only the required file paths, "
                "and do not include code, markdown, or commentary."
            ),
        )
        return self._parse_evaluator_plan(data)

    def _generate_evaluator_file(
        self,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
        plan: EvaluatorPlan,
        file_plan: EvaluatorFilePlan,
        *,
        validation_feedback: str | None = None,
    ) -> dict[str, str]:
        payload = self._evaluator_context_payload(spec, runtime, grounding)
        payload["evaluator_plan"] = plan.to_dict()
        payload["target_file"] = file_plan.to_dict()
        if validation_feedback:
            payload["previous_file_issue"] = validation_feedback

        messages = [
            {"role": "system", "content": self._evaluator_file_system(file_plan.path)},
            {
                "role": "user",
                "content": json.dumps(payload, indent=2),
            },
        ]
        raw = self._bridge.call_llm(messages)
        return {
            "path": file_plan.path,
            "content": self._extract_code_response(raw),
        }

    def _generate_validated_evaluator_file(
        self,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
        plan: EvaluatorPlan,
        file_plan: EvaluatorFilePlan,
        *,
        validation_feedback: str | None = None,
    ) -> dict[str, str]:
        last_issue = validation_feedback
        for attempt in range(1, self.MAX_EVALUATOR_FILE_GENERATION_ATTEMPTS + 1):
            try:
                generated_file = self._generate_evaluator_file(
                    spec,
                    runtime,
                    grounding,
                    plan,
                    file_plan,
                    validation_feedback=last_issue,
                )
                self._validate_generated_file(generated_file)
                return generated_file
            except Exception as exc:  # noqa: BLE001
                last_issue = str(exc)
                self._bridge.post_activity(
                    "error",
                    f"{file_plan.path} generation attempt {attempt} failed: {last_issue}",
                )

        raise ValueError(f"unable to generate {file_plan.path}: {last_issue or 'unknown error'}")

    def _prepare_evaluator(
        self,
        spec: TaskSpec,
        runtime: WorkspaceRuntime,
        grounding: GroundingResult,
    ) -> dict[str, Any]:
        last_issue: str | None = None

        for attempt in range(1, self.MAX_EVALUATOR_GENERATION_ATTEMPTS + 1):
            if attempt > 1:
                self._bridge.post_activity(
                    "status",
                    f"retrying evaluator generation ({attempt}/{self.MAX_EVALUATOR_GENERATION_ATTEMPTS})...",
                )

            try:
                evaluator_plan = self._generate_evaluator_plan(
                    spec,
                    runtime,
                    grounding,
                    validation_feedback=last_issue,
                )
                evaluator_files = []
                for file_plan in evaluator_plan.files:
                    evaluator_files.append(
                        self._generate_validated_evaluator_file(
                            spec,
                            runtime,
                            grounding,
                            evaluator_plan,
                            file_plan,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                last_issue = str(exc)
                self._bridge.post_activity(
                    "error",
                    f"evaluator generation attempt {attempt} failed: {last_issue}",
                )
                continue

            for file_spec in evaluator_files:
                runtime.write_file(file_spec["path"], file_spec["content"])

            baseline = runtime.run_evaluator()
            validation_issue = self._evaluator_validation_issue(spec, baseline)
            if validation_issue is None:
                self._bridge.post_activity(
                    "tool_result",
                    f"evaluator ready: {', '.join(file['path'] for file in evaluator_files)}",
                )
                return {
                    "summary": evaluator_plan.summary,
                    "files": evaluator_files,
                    "baseline": baseline,
                }

            repair_path = self._suggest_evaluator_repair_path(validation_issue)
            if repair_path:
                self._bridge.post_activity(
                    "status",
                    f"repairing {repair_path} after evaluator validation failed",
                )
                try:
                    repaired_file = self._generate_validated_evaluator_file(
                        spec,
                        runtime,
                        grounding,
                        evaluator_plan,
                        self._plan_file_by_path(evaluator_plan, repair_path),
                        validation_feedback=validation_issue,
                    )
                except Exception as exc:  # noqa: BLE001
                    last_issue = str(exc)
                    self._bridge.post_activity(
                        "error",
                        f"evaluator repair failed on attempt {attempt}: {last_issue}",
                    )
                    continue

                runtime.write_file(repaired_file["path"], repaired_file["content"])
                for file_spec in evaluator_files:
                    if file_spec["path"] == repaired_file["path"]:
                        file_spec["content"] = repaired_file["content"]
                        break
                baseline = runtime.run_evaluator()
                validation_issue = self._evaluator_validation_issue(spec, baseline)
                if validation_issue is None:
                    self._bridge.post_activity(
                        "tool_result",
                        f"evaluator ready: {', '.join(file['path'] for file in evaluator_files)}",
                    )
                    return {
                        "summary": evaluator_plan.summary,
                        "files": evaluator_files,
                        "baseline": baseline,
                    }

            last_issue = validation_issue
            self._bridge.post_activity(
                "error",
                f"evaluator validation failed on attempt {attempt}: {validation_issue}",
            )

        raise ValueError(last_issue or "unable to create a valid evaluator")

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
        finish_summary: str | None = None

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
                finish_summary = str(tool_request.get("summary", "")).strip() or None
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
            "finish_summary": finish_summary,
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

        last_error = "tool protocol error"
        for _attempt in range(1, self.MAX_TOOL_REQUEST_RETRIES + 2):
            raw = self._bridge.call_llm(messages)
            try:
                request = self._parse_json_response(raw)
                return self._validate_tool_request(request)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._bridge.post_activity(
                    "error",
                    f"tool response was invalid; requesting corrected JSON ({exc})",
                )
                messages.append({"role": "assistant", "content": raw[:4000]})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was invalid. Reply with exactly one JSON object "
                            "matching the tool_call or finish schema. Do not include markdown, prose, "
                            "or code fences."
                        ),
                    }
                )

        return {
            "type": "finish",
            "summary": f"Stopping iteration because the tool protocol failed: {last_error}",
        }

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
                    "finish_summary": item.get("finish_summary"),
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

    def _finalize_report(self, report: dict[str, Any]) -> dict[str, Any]:
        completed = bool(report.get("best_result", {}).get("all_targets_met", False))
        report["targets_met"] = completed
        report["outcome"] = "targets_met" if completed else "stopped_early"
        problem = self._problem_summary(report)
        if problem:
            report["problem"] = problem
        report["final_summary"] = self._final_summary(report)
        return report

    def _attach_workspace_state(
        self,
        report: dict[str, Any],
        workspace_action: str,
        workspace_initial_entries: list[str],
    ) -> None:
        report["workspace_action"] = workspace_action
        report["workspace_initial_entries"] = list(workspace_initial_entries)
        if "final_summary" in report:
            report["final_summary"] = self._final_summary(report)

    def _final_summary(self, report: dict[str, Any]) -> str:
        best = report.get("best_result", {})
        metrics = best.get("metrics", [])
        targets_met = bool(report.get("targets_met", False))
        lines = ["Automation run finished."]
        lines.append(
            f"Workspace: {self._workspace_action_text(str(report.get('workspace_action', 'empty')))}"
        )
        if targets_met:
            lines.append("Outcome: met all confirmed criteria.")
        else:
            lines.append("Outcome: did not meet all confirmed criteria.")

        lines.append(f"Stop reason: {self._stop_reason_text(str(report.get('stop_reason', 'unknown')))}")

        problem = str(report.get("problem", "")).strip()
        if problem:
            lines.append(f"Problem: {problem}")

        if metrics:
            lines.append("Results:")
            for metric in metrics:
                status = "met" if metric.get("met") else "not met"
                lines.append(
                    f"- {metric.get('name')}: {status} (value={metric.get('value')}, target={metric.get('target')})"
                )

        installed = report.get("installed_packages", [])
        if installed:
            lines.append(f"Installed packages: {', '.join(installed)}")
        lines.append("Artifacts: workspace/.agent/run_report.json")
        return "\n".join(lines)

    def _problem_summary(self, report: dict[str, Any]) -> str:
        stop_reason = str(report.get("stop_reason", "unknown"))
        if stop_reason == "all_targets_met":
            return ""

        reason_text = self._stop_reason_text(stop_reason)
        last_issue = self._last_issue_text(report)
        if last_issue:
            if last_issue.rstrip(".") == reason_text.rstrip("."):
                return last_issue
            return f"{reason_text} Latest issue: {last_issue}"
        return reason_text

    def _last_issue_text(self, report: dict[str, Any]) -> str:
        iterations = report.get("iterations", [])
        for item in reversed(iterations):
            finish_summary = str(item.get("finish_summary") or "").strip()
            if finish_summary:
                return finish_summary
            evaluation = item.get("evaluation", {})
            summary = str(evaluation.get("summary", "")).strip()
            if summary:
                return summary

        best_summary = str(report.get("best_result", {}).get("summary", "")).strip()
        if best_summary:
            return best_summary

        baseline_summary = str(report.get("baseline", {}).get("summary", "")).strip()
        return baseline_summary

    def _stop_reason_text(self, stop_reason: str) -> str:
        mapping = {
            "all_targets_met": "all confirmed metrics were satisfied",
            "plateau": "the run plateaued after consecutive non-improving iterations",
            "iteration_budget": "the run reached the iteration budget",
            "time_limit": "the run reached the time limit",
            "blocked_inputs": "required inputs or assets were missing from the workspace",
            "unsupported_language": "automation currently supports Python only",
            "evaluator_generation_failed": "the agent could not produce a valid evaluator",
        }
        return mapping.get(stop_reason, stop_reason.replace("_", " "))

    def _workspace_action_text(self, workspace_action: str) -> str:
        mapping = {
            "empty": "started from an empty workspace",
            "starter_code": "kept existing files as starter code",
            "cleared": "cleared existing files before the run",
        }
        return mapping.get(workspace_action, workspace_action.replace("_", " "))

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(cleaned[start : end + 1])
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _request_json_object(
        self,
        messages: list[dict[str, str]],
        *,
        correction_prompt: str,
        retries: int = 1,
    ) -> dict[str, Any]:
        attempt_messages = list(messages)
        last_error = "invalid JSON response"
        for _attempt in range(retries + 1):
            raw = self._bridge.call_llm(attempt_messages)
            try:
                return self._parse_json_response(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                attempt_messages.append({"role": "assistant", "content": raw[:4000]})
                attempt_messages.append({"role": "user", "content": correction_prompt})
        raise ValueError(last_error)

    def _parse_evaluator_plan(self, data: dict[str, Any]) -> EvaluatorPlan:
        summary = str(data.get("summary", "")).strip() or "evaluator artifacts planned"
        files = data.get("files")
        if not isinstance(files, list):
            raise ValueError("evaluator plan did not include files")

        file_plans: list[EvaluatorFilePlan] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip().replace("\\", "/")
            purpose = str(item.get("purpose", "")).strip()
            if path.startswith("./"):
                path = path[2:]
            if not path or not purpose:
                continue
            file_plans.append(EvaluatorFilePlan(path=path, purpose=purpose))

        paths = {file.path for file in file_plans}
        required = set(_REQUIRED_EVALUATOR_FILE_PATHS)
        missing = sorted(required - paths)
        extra = sorted(paths - required)
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing {', '.join(missing)}")
            if extra:
                problems.append(f"unexpected {', '.join(extra)}")
            raise ValueError(f"evaluator plan files are invalid: {'; '.join(problems)}")

        ordered_files = [self._plan_file_by_path(EvaluatorPlan(summary, file_plans), path) for path in _REQUIRED_EVALUATOR_FILE_PATHS]
        return EvaluatorPlan(summary=summary, files=ordered_files)

    def _plan_file_by_path(self, plan: EvaluatorPlan, path: str) -> EvaluatorFilePlan:
        for file_plan in plan.files:
            if file_plan.path == path:
                return file_plan
        raise ValueError(f"evaluator plan did not include {path}")

    def _evaluator_file_system(self, path: str) -> str:
        if path == "tests/test_solution.py":
            return _EVALUATOR_TEST_FILE_SYSTEM
        if path == ".agent/evaluator.py":
            return _EVALUATOR_RUNNER_FILE_SYSTEM
        raise ValueError(f"unsupported evaluator file path: {path}")

    def _extract_code_response(self, raw: str) -> str:
        cleaned = raw.strip()
        if not cleaned:
            raise ValueError("generated file content was empty")

        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
            cleaned = cleaned.strip()
        else:
            match = re.search(r"```[^\n]*\n(.*?)\n```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        if not cleaned:
            raise ValueError("generated file content was empty")
        return cleaned

    def _validate_generated_file(self, file_spec: dict[str, str]) -> None:
        path = file_spec["path"]
        content = file_spec["content"]
        if path.startswith("/") or path.startswith("../"):
            raise ValueError(f"generated file path must stay workspace-relative: {path}")
        if path not in _REQUIRED_EVALUATOR_FILE_PATHS:
            raise ValueError(f"generated file path is not allowed: {path}")
        if not content.strip():
            raise ValueError(f"{path} was empty")
        if "/workspace/" in content or "\\workspace\\" in content:
            raise ValueError(f"{path} hard-codes an absolute workspace path")
        if re.search(r"isinstance\s*\([^)]*List\[[^)]*\)", content):
            raise ValueError(f"{path} uses typing generics inside isinstance")
        try:
            compile(content, path, "exec")
        except SyntaxError as exc:
            raise ValueError(
                f"{path} has invalid Python syntax: {exc.msg} (line {exc.lineno})"
            ) from exc

    def _suggest_evaluator_repair_path(self, validation_issue: str) -> str | None:
        lowered = validation_issue.lower()
        if "tests/test_solution.py" in lowered:
            return "tests/test_solution.py"
        if ".agent/evaluator.py" in lowered:
            return ".agent/evaluator.py"
        if lowered.startswith("evaluator execution failed:"):
            return ".agent/evaluator.py"
        if lowered.startswith("evaluator returned invalid json:"):
            return ".agent/evaluator.py"
        if lowered.startswith("evaluator file is missing."):
            return ".agent/evaluator.py"
        if "metric" in lowered:
            return ".agent/evaluator.py"
        return None

    def _evaluator_validation_issue(
        self,
        spec: TaskSpec,
        result: EvaluationResult,
    ) -> str | None:
        summary = result.summary or ""
        if summary.startswith("Evaluator execution failed:"):
            return summary
        if summary.startswith("Evaluator returned invalid JSON:"):
            return summary
        if summary.startswith("Evaluator file is missing."):
            return summary

        if len(result.metrics) != len(spec.metrics):
            return "Evaluator returned a metric list that does not match the confirmed metrics."

        result_by_name = {metric.name: metric for metric in result.metrics}
        for metric_spec in spec.metrics:
            metric = result_by_name.get(metric_spec.name)
            if metric is None:
                return f"Evaluator did not report the confirmed metric '{metric_spec.name}'."
            if metric.kind != metric_spec.kind:
                return (
                    f"Evaluator reported metric '{metric_spec.name}' with kind "
                    f"'{metric.kind}' instead of '{metric_spec.kind}'."
                )
            if not self._metric_targets_match(metric.target, metric_spec.target):
                return (
                    f"Evaluator reported metric '{metric_spec.name}' with target "
                    f"{metric.target} instead of {metric_spec.target}."
                )
        return None

    def _metric_targets_match(
        self,
        actual: float | bool,
        expected: float | bool,
    ) -> bool:
        if isinstance(expected, bool):
            return bool(actual) is expected
        try:
            return abs(float(actual) - float(expected)) < 1e-9
        except (TypeError, ValueError):
            return False

    def _validate_tool_request(self, request: dict[str, Any]) -> dict[str, Any]:
        request_type = str(request.get("type", "")).strip()
        if request_type not in {"tool_call", "finish"}:
            raise ValueError("response type must be 'tool_call' or 'finish'")

        if request_type == "finish":
            return {
                "type": "finish",
                "summary": str(request.get("summary", "")).strip() or "iteration complete",
            }

        tool_name = str(request.get("tool", "")).strip()
        args = request.get("args", {})
        if not tool_name:
            raise ValueError("tool_call responses must include a tool name")
        if not isinstance(args, dict):
            raise ValueError("tool args must be an object")
        if tool_name in {"write_file", "replace_in_file", "remove_path"}:
            path = str(args.get("path", "")).strip()
            if path.startswith(".agent/") or path.startswith("tests/"):
                return {
                    "type": "finish",
                    "summary": "Protected evaluation files should not be modified.",
                }
        return {
            "type": "tool_call",
            "tool": tool_name,
            "args": args,
            "reason": str(request.get("reason", "")).strip(),
        }
