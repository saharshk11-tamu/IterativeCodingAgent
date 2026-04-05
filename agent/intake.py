"""
agent/intake.py - Intake phase of the iterative coding agent.

Responsibilities:
  1. Pre-check: is this a coding task?
  2. Clarification loop: ask targeted questions until the task is concrete.
  3. Metric confirmation: define executable metrics and target values.
  4. Extraction: parse a structured TaskSpec from the conversation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from agent.task_spec import MetricSpec, TaskSpec


@runtime_checkable
class BridgeProtocol(Protocol):
    """Minimum surface IntakeAgent needs from AgentBridge."""

    def ask_user(self, question: str) -> str: ...
    def post_activity(self, kind: str, text: str) -> None: ...
    def post_token(self, token: str) -> None: ...
    def post_message(self, text: str) -> None: ...
    def call_llm(self, messages: list[dict]) -> str: ...


_PRECHECK_SYSTEM = """\
You are a classifier. A user has submitted a request to a coding agent.
Your only job is to decide: is this a coding or programming task?

Reply with exactly one word: YES or NO.

A coding task is any request that requires writing, modifying, debugging,
or explaining code in any programming language - including algorithms,
data structures, APIs, tests, build scripts, SQL, regex, and similar.

A non-coding task is anything else: general knowledge, creative writing,
translation, personal advice, math derivations with no code, etc."""

_CLARIFICATION_SYSTEM = """\
You are the intake phase of a coding agent. Your goal is to gather exactly
enough information to write and evaluate code - no more, no less.

Given the conversation so far, decide if there is ONE critical piece of
information still missing that would materially affect the implementation.

If you need more information, ask EXACTLY ONE short, specific question.
Format your response as:
  QUESTION: <your question here>

If you have enough information, respond with exactly:
  SUFFICIENT

Rules:
- Do NOT ask about things that can be reasonably assumed.
- Do NOT ask about edge cases that can be handled with sensible defaults.
- Only ask about genuine ambiguities that would cause fundamentally different
  implementations or evaluation strategies.
- Never ask more than one question at a time."""

_SUFFICIENCY_SYSTEM = """\
You are reviewing whether a coding task description is complete enough to
begin implementation. Answer YES if all of the following are known or can
be reasonably assumed:
  - What the code should do
  - The programming language (or a safe default exists)
  - The primary inputs and outputs
  - Any hard constraints
  - What success will be measured against

Answer NO if any of these are genuinely unclear.

Reply with exactly one word: YES or NO."""

_METRICS_GENERATION_SYSTEM = """\
You are defining executable success metrics for a coding task.

Given the conversation, produce a JSON array where each item has:
  - name: short metric name
  - description: one sentence describing what is measured
  - kind: one of "pass_fail", "maximize", "minimize"
  - target: boolean for pass_fail, number for maximize/minimize

Rules:
- Every metric must be machine-checkable.
- Include numeric target values when relevant, such as accuracy thresholds,
  latency ceilings, throughput minima, or memory limits.
- Prefer 1 to 4 metrics total.
- Make the first metric the primary optimization target.
- If the user did not specify an exact numeric threshold, propose a reasonable
  default that they can confirm or revise.
- Output ONLY the JSON array."""

_METRICS_REVISION_SYSTEM = """\
You are revising a JSON array of executable metrics based on user feedback.

Keep the same schema:
  - name
  - description
  - kind
  - target

Apply the user's requested changes exactly and output ONLY the revised JSON array.
Do not include markdown fences or explanations."""

_EXTRACTION_SYSTEM = """\
You are extracting a structured task specification from a conversation.

Produce a JSON object with exactly these fields:
{
  "refined_description": "<one or two sentence canonical task statement>",
  "language": "<programming language, lowercase, e.g. python>",
  "task_type": "<one of: function | class | script | api_endpoint | data_pipeline | other_coding>",
  "requirements": ["<functional requirement sentence>", ...],
  "constraints": ["<non-functional constraint sentence>", ...],
  "dependencies": ["<library name>", ...],
  "examples": [{"input": "<value>", "output": "<value>"}, ...]
}

Rules:
- requirements: functional requirements only
- constraints: non-functional requirements only
- dependencies: only libraries explicitly mentioned by the user
- examples: only user-provided input/output examples
- Output ONLY the JSON object."""

_CLARIFICATION_TRANSITION_SYSTEM = """\
You are the friendly face of a coding agent. The user just submitted a coding
request. Write a single short sentence (max 15 words) that:
  - acknowledges the request
  - sets the expectation that you may ask a clarifying question

Do NOT ask the question. Do NOT use markdown. Output only the sentence."""

_METRICS_TRANSITION_SYSTEM = """\
You are the friendly face of a coding agent. You have just finished clarifying
a coding task and are about to propose executable success metrics.
Write a single short sentence (max 15 words) that:
  - signals you understand the task
  - introduces the metric proposal naturally

Do NOT list the metrics. Do NOT use markdown. Output only the sentence."""

_REFUSAL_MESSAGE = (
    "I can only help with coding and programming tasks. "
    "Please describe a software problem such as writing a function, "
    "debugging code, designing a class, or implementing an algorithm."
)

_CODING_SIGNALS = {
    "write", "implement", "code", "function", "class", "script",
    "algorithm", "sort", "parse", "api", "endpoint", "test",
    "debug", "fix", "refactor", "build", "program", "module",
    "loop", "recursion", "data structure", "sql", "regex", "cli",
    "library", "package", "compile", "syntax", "runtime", "async",
    "thread", "coroutine", "decorator", "iterator", "generator",
    "type hint", "interface", "inherit", "polymorphism", "http",
    "json", "yaml", "csv", "database", "orm", "query",
}

_NON_CODING_SIGNALS = {
    "tell me about", "who is", "what is the history of",
    "explain the concept", "write a poem", "write an essay",
    "recipe", "weather", "news", "joke", "translate",
    "summarize this article", "what happened", "give me advice",
    "recommend a", "what do you think about",
}


class IntakeAgent:
    MAX_QUESTIONS: int = 4

    def __init__(self, bridge: BridgeProtocol) -> None:
        self._bridge = bridge
        self._history: list[dict[str, str]] = []
        self._confirmed_metrics: list[MetricSpec] = []

    def run(self, user_prompt: str) -> TaskSpec | None:
        self._history = []
        self._confirmed_metrics = []

        self._bridge.post_activity("status", "checking request type...")
        if not self._is_coding_task(user_prompt):
            self._bridge.post_token(_REFUSAL_MESSAGE)
            return None

        self._history.append({"role": "user", "content": user_prompt})

        self._bridge.post_activity("status", "analyzing request...")
        self._bridge.post_message(
            self._generate_transition(user_prompt, _CLARIFICATION_TRANSITION_SYSTEM)
        )
        turns = self._run_clarification_loop()

        self._bridge.post_activity("status", "defining executable metrics...")
        self._bridge.post_message(
            self._generate_transition(user_prompt, _METRICS_TRANSITION_SYSTEM)
        )
        self._confirm_metrics()

        self._bridge.post_activity("status", "finalizing task specification...")
        spec = self._extract_task_spec(user_prompt, clarification_turns=turns)

        self._bridge.post_activity(
            "status",
            f"intake complete - {spec.task_type} / {spec.language} / {turns} clarification turn(s)",
        )
        return spec

    def _generate_transition(self, user_prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        return self._bridge.call_llm(messages).strip()

    def _is_coding_task(self, prompt: str) -> bool:
        result = self._heuristic_check(prompt)
        if result is not None:
            return result
        self._bridge.post_activity("status", "classifying request...")
        return self._llm_precheck(prompt)

    def _heuristic_check(self, prompt: str) -> bool | None:
        lower = prompt.lower()
        coding_hits = sum(1 for signal in _CODING_SIGNALS if signal in lower)
        non_coding_hits = sum(1 for signal in _NON_CODING_SIGNALS if signal in lower)

        if coding_hits >= 2 and non_coding_hits == 0:
            return True
        if non_coding_hits >= 1 and coding_hits == 0:
            return False
        return None

    def _llm_precheck(self, prompt: str) -> bool:
        messages = [
            {"role": "system", "content": _PRECHECK_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self._bridge.call_llm(messages).strip()
        return response.upper().startswith("Y")

    def _run_clarification_loop(self) -> int:
        turns = 0
        for _ in range(self.MAX_QUESTIONS):
            question = self._generate_next_question(turns)
            if question is None:
                break

            answer = self._bridge.ask_user(question)
            turns += 1
            self._history.append({"role": "assistant", "content": question})
            self._history.append({"role": "user", "content": answer})

            if self._is_sufficient():
                break

        return turns

    def _generate_next_question(self, questions_asked: int) -> str | None:
        system = (
            _CLARIFICATION_SYSTEM
            + f"\n\nQuestions asked so far: {questions_asked} of {self.MAX_QUESTIONS} allowed."
        )
        messages = [{"role": "system", "content": system}] + self._history
        response = self._bridge.call_llm(messages).strip()

        if "SUFFICIENT" in response.upper() and "QUESTION:" not in response.upper():
            return None

        match = re.search(r"QUESTION:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        if "?" in response:
            return response
        return None

    def _is_sufficient(self) -> bool:
        messages = [
            {"role": "system", "content": _SUFFICIENCY_SYSTEM},
            {
                "role": "user",
                "content": "Conversation:\n"
                + "\n".join(
                    f"{message['role'].upper()}: {message['content']}"
                    for message in self._history
                ),
            },
        ]
        response = self._bridge.call_llm(messages).strip()
        return response.upper().startswith("Y")

    def _confirm_metrics(self) -> None:
        conversation_text = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in self._history
        )
        messages = [
            {"role": "system", "content": _METRICS_GENERATION_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{conversation_text}"},
        ]
        metrics = self._parse_metrics(self._bridge.call_llm(messages))

        while True:
            answer = self._bridge.ask_user(
                "Here are the proposed executable success metrics and target values:\n\n"
                f"{self._format_metrics(metrics)}\n\n"
                "Do these look correct? (yes / no - or describe what to change)"
            )

            if answer.strip().lower().startswith("y"):
                break

            feedback = answer.strip()
            if feedback.lower() in {"no", "n", "nope", "nah"}:
                feedback = self._bridge.ask_user(
                    "What would you like to add, remove, or change about the metrics or targets?"
                )

            self._bridge.post_activity("status", "revising confirmed metrics...")
            revision_messages = [
                {"role": "system", "content": _METRICS_REVISION_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "Current metrics JSON:\n"
                        f"{json.dumps([metric.to_dict() for metric in metrics], indent=2)}\n\n"
                        f"Requested changes: {feedback}"
                    ),
                },
            ]
            metrics = self._parse_metrics(self._bridge.call_llm(revision_messages))

        self._confirmed_metrics = metrics
        self._history.append(
            {
                "role": "assistant",
                "content": (
                    "Here are the confirmed executable success metrics:\n"
                    f"{self._format_metrics(metrics)}"
                ),
            }
        )
        self._history.append(
            {"role": "user", "content": "Yes, those metrics and targets look correct."}
        )

    def _parse_metrics(self, raw: str) -> list[MetricSpec]:
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            self._bridge.post_activity(
                "error",
                "metric parsing failed; using a fallback pass/fail metric",
            )
            return [
                MetricSpec(
                    name="task_completion",
                    description="Implementation satisfies the task requirements",
                    kind="pass_fail",
                    target=True,
                    priority=1,
                )
            ]

        if not isinstance(data, list) or not data:
            return [
                MetricSpec(
                    name="task_completion",
                    description="Implementation satisfies the task requirements",
                    kind="pass_fail",
                    target=True,
                    priority=1,
                )
            ]

        metrics: list[MetricSpec] = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            metrics.append(MetricSpec.from_dict(item, priority=index))

        if not metrics:
            metrics.append(
                MetricSpec(
                    name="task_completion",
                    description="Implementation satisfies the task requirements",
                    kind="pass_fail",
                    target=True,
                    priority=1,
                )
            )
        return metrics

    def _format_metrics(self, metrics: list[MetricSpec]) -> str:
        lines = []
        for metric in metrics:
            lines.append(
                f"{metric.priority}. {metric.name} [{metric.kind}] - {metric.description} (target {metric.target_text()})"
            )
        return "\n".join(lines)

    def _extract_task_spec(
        self, original_prompt: str, clarification_turns: int
    ) -> TaskSpec:
        conversation_text = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in self._history
        )
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{conversation_text}"},
        ]
        raw = self._bridge.call_llm(messages).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            data = json.loads(raw)
            return TaskSpec(
                original_prompt=original_prompt,
                refined_description=data.get("refined_description", original_prompt),
                language=data.get("language", "python"),
                task_type=data.get("task_type", "other_coding"),
                requirements=data.get("requirements", []),
                constraints=data.get("constraints", []),
                dependencies=data.get("dependencies", []),
                examples=data.get("examples", []),
                metrics=self._confirmed_metrics,
                clarification_turns=clarification_turns,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            self._bridge.post_activity(
                "error", f"spec extraction parse error ({exc}); using fallback"
            )
            return TaskSpec(
                original_prompt=original_prompt,
                refined_description=original_prompt,
                language="python",
                task_type="other_coding",
                metrics=self._confirmed_metrics,
                clarification_turns=clarification_turns,
            )
