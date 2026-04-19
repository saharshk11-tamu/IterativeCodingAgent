"""
Benchmark harness for IterativeCodingAgent.

Bypasses the intake / UI entirely — constructs TaskSpec objects directly and
drives AutomationOrchestrator headlessly.  Results are written to
benchmark_results/<timestamp>.json and summarised in the terminal.

Usage:
    python benchmark.py \\
        --provider gemini --model gemini-2.0-flash --api-key $GEMINI_API_KEY

    python benchmark.py \\
        --provider anthropic --model claude-sonnet-4-6 --api-key $ANTHROPIC_API_KEY \\
        --tasks fibonacci palindrome http_calc --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is on the path when running as a top-level script.
sys.path.insert(0, str(Path(__file__).parent))

from agent.automation import AutomationOrchestrator
from agent.intake import BridgeProtocol
from agent.task_spec import MetricSpec, TaskSpec
from llm import LLMConfig, generate_text


# ---------------------------------------------------------------------------
# Canonical task suite
# ---------------------------------------------------------------------------

def _metric(name: str, kind: str, target: float, priority: int, description: str) -> MetricSpec:
    return MetricSpec(
        name=name,
        description=description,
        kind=kind,  # type: ignore[arg-type]
        target=target,
        priority=priority,
    )


TASK_SUITE: dict[str, TaskSpec] = {
    "fibonacci": TaskSpec(
        original_prompt="Write a Python function that returns the nth Fibonacci number.",
        refined_description=(
            "Implement a Python function fibonacci(n) that returns the nth Fibonacci number "
            "(0-indexed, fibonacci(0)=0, fibonacci(1)=1). Handle edge cases gracefully."
        ),
        language="python",
        task_type="function",
        requirements=[
            "Function named fibonacci(n) that accepts a non-negative integer.",
            "Returns the nth Fibonacci number.",
            "Handles n=0 and n=1 correctly.",
        ],
        constraints=["No external libraries."],
        metrics=[
            _metric("test_coverage", "maximize", 0.9, 1,
                    "Fraction of code lines exercised by the test suite"),
            _metric("correctness", "pass_fail", True, 2,
                    "All test cases pass including edge cases"),
        ],
    ),

    "palindrome": TaskSpec(
        original_prompt="Write a Python function that checks if a string is a palindrome.",
        refined_description=(
            "Implement is_palindrome(s: str) -> bool that returns True if s is a palindrome "
            "(ignoring case and non-alphanumeric characters), False otherwise."
        ),
        language="python",
        task_type="function",
        requirements=[
            "Function named is_palindrome(s) accepting a string.",
            "Case-insensitive comparison.",
            "Ignores non-alphanumeric characters.",
        ],
        constraints=["No external libraries."],
        metrics=[
            _metric("test_coverage", "maximize", 0.9, 1,
                    "Fraction of code lines exercised by the test suite"),
            _metric("correctness", "pass_fail", True, 2,
                    "All test cases pass including edge cases"),
        ],
    ),

    "binary_search": TaskSpec(
        original_prompt="Write a Python function that performs binary search on a sorted list.",
        refined_description=(
            "Implement binary_search(arr: list, target) -> int that returns the index of "
            "target in the sorted list arr, or -1 if not found."
        ),
        language="python",
        task_type="function",
        requirements=[
            "Function named binary_search(arr, target).",
            "Returns the index of target or -1 if not present.",
            "Input list is sorted in ascending order.",
        ],
        constraints=["Must use binary search algorithm (O(log n)), not linear scan."],
        metrics=[
            _metric("test_coverage", "maximize", 0.9, 1,
                    "Fraction of code lines exercised by the test suite"),
            _metric("correctness", "pass_fail", True, 2,
                    "All test cases pass"),
        ],
    ),

    "http_calc": TaskSpec(
        original_prompt=(
            "Create an HTTP server in Python with endpoints for basic arithmetic: "
            "addition, subtraction, multiplication, division."
        ),
        refined_description=(
            "Implement a Python HTTP server with four endpoints: /add, /subtract, /multiply, "
            "/divide. Each accepts query params a and b (floats) and returns JSON {\"result\": ...}. "
            "Division by zero should return HTTP 400."
        ),
        language="python",
        task_type="api_endpoint",
        requirements=[
            "Endpoints: GET /add?a=&b=, GET /subtract?a=&b=, GET /multiply?a=&b=, GET /divide?a=&b=",
            "Returns JSON {\"result\": <number>}.",
            "Division by zero returns HTTP 400 with {\"error\": \"division by zero\"}.",
            "Missing or non-numeric parameters return HTTP 400.",
        ],
        constraints=["Response time under 500 ms.", "Error rate under 1%."],
        metrics=[
            _metric("test_coverage", "maximize", 0.85, 1,
                    "Percentage of code lines executed by tests"),
            _metric("response_time", "minimize", 500.0, 2,
                    "Maximum response time in milliseconds"),
            _metric("error_rate", "minimize", 0.01, 3,
                    "Fraction of requests that result in an unexpected error"),
        ],
    ),

    "word_count": TaskSpec(
        original_prompt="Write a Python script that counts word frequencies in a text file.",
        refined_description=(
            "Implement a Python function word_frequencies(text: str) -> dict[str, int] that "
            "returns a dictionary mapping each word (lowercase, stripped of punctuation) to "
            "its frequency in the input text."
        ),
        language="python",
        task_type="function",
        requirements=[
            "Function named word_frequencies(text) accepting a string.",
            "Returns a dict mapping lowercase words to their integer counts.",
            "Strips leading/trailing punctuation from words.",
            "Handles empty string input.",
        ],
        constraints=["No external libraries."],
        metrics=[
            _metric("test_coverage", "maximize", 0.9, 1,
                    "Fraction of code lines exercised by the test suite"),
            _metric("correctness", "pass_fail", True, 2,
                    "All test cases pass"),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Headless bridge
# ---------------------------------------------------------------------------

class HeadlessBridge:
    """
    BridgeProtocol implementation that runs without a UI.

    Records all activity events, counts LLM calls, and timestamps each
    phase transition so we can compute per-phase wall-clock times.
    """

    def __init__(self, config: LLMConfig, verbose: bool = False) -> None:
        self._config = config
        self._verbose = verbose
        self.llm_call_count: int = 0
        self.events: list[dict[str, Any]] = []
        self.phase_times: dict[str, float] = {}
        self._phase_start: dict[str, float] = {}

    # -- BridgeProtocol -------------------------------------------------------

    def ask_user(self, question: str) -> str:
        self._record("ask_user", question)
        # Auto-answer: clear workspace so each run starts fresh.
        return "clear"

    def post_activity(self, kind: str, text: str) -> None:
        self._record(kind, text)
        if self._verbose:
            print(f"  [{kind}] {text[:120]}")
        # Track phase transitions for timing.
        self._update_phase_timing(kind, text)

    def post_token(self, token: str) -> None:
        pass

    def post_message(self, text: str) -> None:
        self._record("message", text)

    def call_llm(self, messages: list[dict]) -> str:
        self.llm_call_count += 1
        return generate_text(self._config, messages).text

    # -- Helpers --------------------------------------------------------------

    def _record(self, kind: str, text: str) -> None:
        self.events.append({"ts": time.monotonic(), "kind": kind, "text": text})

    def _update_phase_timing(self, kind: str, text: str) -> None:
        now = time.monotonic()
        lower = text.lower()

        if kind == "status" and "grounding" in lower:
            self._phase_start["grounding"] = now
        elif kind == "status" and "generating evaluator" in lower:
            if "grounding" in self._phase_start and "grounding" not in self.phase_times:
                self.phase_times["grounding_s"] = now - self._phase_start["grounding"]
            self._phase_start["evaluator_gen"] = now
        elif kind == "tool_result" and "evaluator ready" in lower:
            if "evaluator_gen" in self._phase_start:
                self.phase_times["evaluator_gen_s"] = now - self._phase_start["evaluator_gen"]
        elif kind == "status" and "starting iteration" in lower:
            if "evaluator_gen" in self._phase_start and "evaluator_gen_s" not in self.phase_times:
                self.phase_times["evaluator_gen_s"] = now - self._phase_start["evaluator_gen"]
            if "iterations" not in self._phase_start:
                self._phase_start["iterations"] = now
        elif kind == "status" and "automation stopped" in lower:
            if "iterations" in self._phase_start:
                self.phase_times["iterations_s"] = now - self._phase_start["iterations"]

    def evaluator_gen_attempts(self) -> int:
        """Count how many outer evaluator-generation attempts were made."""
        return sum(
            1 for e in self.events
            if e["kind"] == "status" and "retrying evaluator generation" in e["text"].lower()
        ) + 1  # +1 for the first (non-retry) attempt

    def evaluator_gen_succeeded_first_attempt(self) -> bool:
        return self.evaluator_gen_attempts() == 1 and any(
            e["kind"] == "tool_result" and "evaluator ready" in e["text"].lower()
            for e in self.events
        )


# ---------------------------------------------------------------------------
# Single-run execution
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    task_id: str
    run_index: int
    success: bool           # all_targets_met
    stop_reason: str
    total_s: float
    grounding_s: float
    evaluator_gen_s: float
    iterations_s: float
    iteration_count: int
    llm_calls: int
    evaluator_gen_attempts: int
    evaluator_gen_first_attempt: bool
    baseline_metrics: list[dict[str, Any]]
    final_metrics: list[dict[str, Any]]
    error: str | None = None


def run_single(
    task_id: str,
    spec: TaskSpec,
    config: LLMConfig,
    workspace_root: str,
    run_index: int,
    verbose: bool = False,
) -> RunResult:
    bridge = HeadlessBridge(config, verbose=verbose)
    t_start = time.monotonic()
    error: str | None = None
    report: dict[str, Any] = {}

    try:
        orchestrator = AutomationOrchestrator(bridge)
        report = orchestrator.run(spec, workspace_dir=workspace_root)
    except Exception as exc:
        error = str(exc)
        if verbose:
            print(f"  [ERROR] run failed: {exc}")

    t_end = time.monotonic()
    total_s = t_end - t_start

    stop_reason = str(report.get("stop_reason", "error"))
    success = bool(report.get("targets_met", False))
    iterations = report.get("iterations", [])
    baseline_metrics = report.get("baseline", {}).get("metrics", [])
    final_metrics = report.get("best_result", {}).get("metrics", [])

    return RunResult(
        task_id=task_id,
        run_index=run_index,
        success=success,
        stop_reason=stop_reason,
        total_s=total_s,
        grounding_s=bridge.phase_times.get("grounding_s", 0.0),
        evaluator_gen_s=bridge.phase_times.get("evaluator_gen_s", 0.0),
        iterations_s=bridge.phase_times.get("iterations_s", 0.0),
        iteration_count=len(iterations),
        llm_calls=bridge.llm_call_count,
        evaluator_gen_attempts=bridge.evaluator_gen_attempts(),
        evaluator_gen_first_attempt=bridge.evaluator_gen_succeeded_first_attempt(),
        baseline_metrics=baseline_metrics,
        final_metrics=final_metrics,
        error=error,
    )


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

@dataclass
class TaskAggregate:
    task_id: str
    runs: int
    pass_at_1: float        # % of runs that succeeded
    mean_iterations: float  # across all runs
    mean_total_s: float
    mean_grounding_s: float
    mean_evaluator_gen_s: float
    mean_iterations_s: float
    mean_llm_calls: float
    evaluator_gen_first_attempt_rate: float  # % where gen succeeded on first outer attempt
    mean_evaluator_gen_attempts: float
    metric_improvement: dict[str, float]    # metric_name -> avg (final-baseline)/target_distance
    run_results: list[RunResult] = field(default_factory=list)


def aggregate(results: list[RunResult]) -> TaskAggregate:
    n = len(results)
    assert n > 0

    task_id = results[0].task_id

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    pass_at_1 = avg([1.0 if r.success else 0.0 for r in results])
    mean_iterations = avg([float(r.iteration_count) for r in results])
    mean_total_s = avg([r.total_s for r in results])
    mean_grounding_s = avg([r.grounding_s for r in results])
    mean_evaluator_gen_s = avg([r.evaluator_gen_s for r in results])
    mean_iterations_s = avg([r.iterations_s for r in results])
    mean_llm_calls = avg([float(r.llm_calls) for r in results])
    eva_first_rate = avg([1.0 if r.evaluator_gen_first_attempt else 0.0 for r in results])
    mean_eva_attempts = avg([float(r.evaluator_gen_attempts) for r in results])

    # Metric improvement: per metric, average normalised improvement across runs.
    metric_improvement: dict[str, dict[str, list[float]]] = {}
    for r in results:
        baseline_by_name = {m["name"]: m for m in r.baseline_metrics}
        for m in r.final_metrics:
            name = m["name"]
            kind = m.get("kind", "pass_fail")
            target = m.get("target", 1.0)
            final_val = m.get("value", 0.0)
            base_val = baseline_by_name.get(name, {}).get("value", final_val)
            if name not in metric_improvement:
                metric_improvement[name] = {"improvements": [], "kind": [kind], "target": [target]}
            if kind == "pass_fail":
                imp = 1.0 if bool(final_val) else 0.0
            elif kind == "maximize":
                try:
                    denom = float(target) - float(base_val) if float(target) != float(base_val) else 1.0
                    imp = (float(final_val) - float(base_val)) / abs(denom)
                except (TypeError, ValueError, ZeroDivisionError):
                    imp = 0.0
            else:  # minimize
                try:
                    denom = float(base_val) - float(target) if float(base_val) != float(target) else 1.0
                    imp = (float(base_val) - float(final_val)) / abs(denom)
                except (TypeError, ValueError, ZeroDivisionError):
                    imp = 0.0
            metric_improvement[name]["improvements"].append(imp)

    avg_improvement = {
        name: avg(data["improvements"])
        for name, data in metric_improvement.items()
    }

    return TaskAggregate(
        task_id=task_id,
        runs=n,
        pass_at_1=pass_at_1,
        mean_iterations=mean_iterations,
        mean_total_s=mean_total_s,
        mean_grounding_s=mean_grounding_s,
        mean_evaluator_gen_s=mean_evaluator_gen_s,
        mean_iterations_s=mean_iterations_s,
        mean_llm_calls=mean_llm_calls,
        evaluator_gen_first_attempt_rate=eva_first_rate,
        mean_evaluator_gen_attempts=mean_eva_attempts,
        metric_improvement=avg_improvement,
        run_results=results,
    )


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _time(s: float) -> str:
    if s < 0.5:
        return "—"
    if s < 60:
        return f"{s:.0f}s"
    return f"{s / 60:.1f}m"


def print_summary(aggregates: list[TaskAggregate], provider: str, model: str) -> None:
    divider = "─" * 90
    print(f"\n{'BENCHMARK RESULTS':^90}")
    print(f"{'Provider: ' + provider + '  Model: ' + model:^90}")
    print(divider)

    header = (
        f"{'Task':<18} {'Runs':>4} {'Pass@1':>7} {'Iters':>6} "
        f"{'Total':>7} {'EvalGen':>8} {'1stAttempt':>11} {'LLM calls':>10}"
    )
    print(header)
    print(divider)

    for agg in aggregates:
        print(
            f"{agg.task_id:<18} {agg.runs:>4} {_pct(agg.pass_at_1):>7} "
            f"{agg.mean_iterations:>6.1f} {_time(agg.mean_total_s):>7} "
            f"{_time(agg.mean_evaluator_gen_s):>8} {_pct(agg.evaluator_gen_first_attempt_rate):>11} "
            f"{agg.mean_llm_calls:>10.1f}"
        )

    print(divider)

    # Per-task metric improvement table
    print(f"\n{'METRIC IMPROVEMENT (avg normalised delta from baseline toward target)':^90}")
    print(divider)
    for agg in aggregates:
        if agg.metric_improvement:
            parts = "  ".join(
                f"{name}: {v:+.2f}" for name, v in agg.metric_improvement.items()
            )
            print(f"  {agg.task_id:<18} {parts}")
    print(divider)

    # Per-run detail
    print(f"\n{'PER-RUN DETAIL':^90}")
    print(divider)
    detail_header = (
        f"  {'Task':<18} {'Run':>3} {'OK':>3} {'Stop reason':<22} "
        f"{'Total':>7} {'Iters':>6} {'LLM':>5}"
    )
    print(detail_header)
    print(divider)
    for agg in aggregates:
        for r in agg.run_results:
            ok = "Y" if r.success else "N"
            stop = r.stop_reason[:21]
            err = f" [{r.error[:30]}]" if r.error else ""
            print(
                f"  {r.task_id:<18} {r.run_index:>3} {ok:>3} {stop:<22} "
                f"{_time(r.total_s):>7} {r.iteration_count:>6} {r.llm_calls:>5}{err}"
            )
    print(divider)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def save_results(
    aggregates: list[TaskAggregate],
    config: LLMConfig,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"benchmark_{ts}.json"

    payload = {
        "provider": config.provider,
        "model": config.model,
        "timestamp": ts,
        "aggregates": [
            {
                "task_id": agg.task_id,
                "runs": agg.runs,
                "pass_at_1": agg.pass_at_1,
                "mean_iterations": agg.mean_iterations,
                "mean_total_s": agg.mean_total_s,
                "mean_grounding_s": agg.mean_grounding_s,
                "mean_evaluator_gen_s": agg.mean_evaluator_gen_s,
                "mean_iterations_s": agg.mean_iterations_s,
                "mean_llm_calls": agg.mean_llm_calls,
                "evaluator_gen_first_attempt_rate": agg.evaluator_gen_first_attempt_rate,
                "mean_evaluator_gen_attempts": agg.mean_evaluator_gen_attempts,
                "metric_improvement": agg.metric_improvement,
                "runs_detail": [
                    {
                        "run_index": r.run_index,
                        "success": r.success,
                        "stop_reason": r.stop_reason,
                        "total_s": r.total_s,
                        "grounding_s": r.grounding_s,
                        "evaluator_gen_s": r.evaluator_gen_s,
                        "iterations_s": r.iterations_s,
                        "iteration_count": r.iteration_count,
                        "llm_calls": r.llm_calls,
                        "evaluator_gen_attempts": r.evaluator_gen_attempts,
                        "evaluator_gen_first_attempt": r.evaluator_gen_first_attempt,
                        "baseline_metrics": r.baseline_metrics,
                        "final_metrics": r.final_metrics,
                        "error": r.error,
                    }
                    for r in agg.run_results
                ],
            }
            for agg in aggregates
        ],
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark harness for IterativeCodingAgent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "ollama"],
        default=os.environ.get("BENCHMARK_PROVIDER", "gemini"),
        help="LLM provider (default: gemini or $BENCHMARK_PROVIDER)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("BENCHMARK_MODEL", ""),
        help="Model ID (default: $BENCHMARK_MODEL or provider default)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BENCHMARK_API_KEY", ""),
        help="API key (default: $BENCHMARK_API_KEY)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_SUITE.keys()) + ["all"],
        default=["all"],
        help=f"Tasks to run (default: all). Choices: {list(TASK_SUITE.keys())}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per task (default: 1; use 3+ for statistical significance)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for JSON results (default: benchmark_results/)",
    )
    parser.add_argument(
        "--workspace-base",
        default="",
        help="Base dir for per-run workspaces (default: system temp dir)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print activity events as they happen",
    )
    return parser.parse_args()


_PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",
}


def main() -> None:
    args = parse_args()

    model = args.model or _PROVIDER_DEFAULT_MODELS.get(args.provider, "")
    if not model:
        print(f"ERROR: --model is required for provider '{args.provider}'", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key
    if not api_key and args.provider != "ollama":
        print("ERROR: --api-key (or $BENCHMARK_API_KEY) is required", file=sys.stderr)
        sys.exit(1)

    config = LLMConfig(provider=args.provider, model=model, api_key=api_key)

    task_ids = list(TASK_SUITE.keys()) if "all" in args.tasks else args.tasks
    output_dir = Path(args.output_dir)
    workspace_base = args.workspace_base or None

    print(f"\nBenchmark: provider={args.provider} model={model}")
    print(f"Tasks: {task_ids}  Runs per task: {args.runs}\n")

    all_results: list[RunResult] = []

    for task_id in task_ids:
        spec = TASK_SUITE[task_id]
        print(f"{'─' * 60}")
        print(f"Task: {task_id}  ({args.runs} run(s))")

        for run_idx in range(1, args.runs + 1):
            ws_dir = tempfile.mkdtemp(
                prefix=f"bench_{task_id}_r{run_idx}_",
                dir=workspace_base,
            )
            try:
                print(f"  run {run_idx}/{args.runs}  workspace={ws_dir}")
                result = run_single(
                    task_id=task_id,
                    spec=spec,
                    config=config,
                    workspace_root=ws_dir,
                    run_index=run_idx,
                    verbose=args.verbose,
                )
                all_results.append(result)
                status = "PASS" if result.success else f"FAIL ({result.stop_reason})"
                print(
                    f"  run {run_idx}: {status}  "
                    f"{result.total_s:.0f}s  "
                    f"{result.iteration_count} iter(s)  "
                    f"{result.llm_calls} LLM calls"
                )
            finally:
                shutil.rmtree(ws_dir, ignore_errors=True)

    # Aggregate and display.
    aggregates: list[TaskAggregate] = []
    for task_id in task_ids:
        task_runs = [r for r in all_results if r.task_id == task_id]
        if task_runs:
            aggregates.append(aggregate(task_runs))

    print_summary(aggregates, provider=args.provider, model=model)

    out_path = save_results(aggregates, config, output_dir)
    print(f"\nResults saved to: {out_path}\n")


if __name__ == "__main__":
    main()
