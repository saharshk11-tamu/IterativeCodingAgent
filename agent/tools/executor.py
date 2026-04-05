"""Tool-call dispatcher for the autonomous workspace workflow."""

from __future__ import annotations

from typing import Any

from agent.evaluation import EvaluationResult
from agent.tools.runtime import ToolOutcome, WorkspaceRuntime


def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    runtime: WorkspaceRuntime,
) -> tuple[ToolOutcome, EvaluationResult | None]:
    try:
        if tool_name == "list_files":
            return runtime.list_files(str(args.get("path", "."))), None
        if tool_name == "read_file":
            return runtime.read_file(str(args["path"])), None
        if tool_name == "write_file":
            return runtime.write_file(str(args["path"]), str(args.get("content", ""))), None
        if tool_name == "replace_in_file":
            return (
                runtime.replace_in_file(
                    str(args["path"]),
                    str(args.get("search", "")),
                    str(args.get("replace", "")),
                ),
                None,
            )
        if tool_name == "remove_path":
            return runtime.remove_path(str(args["path"])), None
        if tool_name == "install_python_packages":
            specs = args.get("specs", [])
            if not isinstance(specs, list):
                return ToolOutcome(False, "specs must be a list"), None
            return runtime.install_python_packages([str(spec) for spec in specs]), None
        if tool_name == "run_python_tests":
            paths = args.get("paths")
            if paths is None:
                return runtime.run_python_tests(None), None
            if not isinstance(paths, list):
                return ToolOutcome(False, "paths must be a list or null"), None
            return runtime.run_python_tests([str(path) for path in paths]), None
        if tool_name == "run_python_script":
            path = str(args["path"])
            raw_args = args.get("args", [])
            if not isinstance(raw_args, list):
                return ToolOutcome(False, "args must be a list"), None
            return runtime.run_python_script(path, [str(arg) for arg in raw_args]), None
        if tool_name == "run_evaluator":
            result = runtime.run_evaluator()
            return (
                ToolOutcome(True, result.summary or "Evaluator completed", result.to_dict()),
                result,
            )
    except Exception as exc:  # noqa: BLE001
        return ToolOutcome(False, f"{tool_name} failed: {exc}"), None

    return ToolOutcome(False, f"Unknown tool: {tool_name}"), None
