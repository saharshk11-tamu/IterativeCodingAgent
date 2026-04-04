"""
agent/intake.py — Intake phase of the iterative coding agent.

Responsibilities:
  1. Pre-check: is this a coding task? Refuse otherwise.
  2. Clarification loop: ask targeted questions until we have enough context.
  3. Extraction: parse a structured TaskSpec from the conversation.

This module has zero imports from cli/ — it communicates exclusively through
the BridgeProtocol interface, keeping the dependency graph a clean DAG.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from agent.task_spec import TaskSpec


# ── Bridge protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class BridgeProtocol(Protocol):
    """Minimum surface IntakeAgent needs from AgentBridge."""

    def ask_user(self, question: str) -> str: ...
    def post_activity(self, kind: str, text: str) -> None: ...
    def post_token(self, token: str) -> None: ...
    def post_message(self, text: str) -> None: ...
    def call_llm(self, messages: list[dict]) -> str: ...


# ── Prompts ───────────────────────────────────────────────────────────────────

_PRECHECK_SYSTEM = """\
You are a classifier. A user has submitted a request to a coding agent.
Your only job is to decide: is this a coding or programming task?

Reply with exactly one word: YES or NO.

A coding task is any request that requires writing, modifying, debugging,
or explaining code in any programming language — including algorithms,
data structures, APIs, tests, build scripts, SQL, regex, and similar.

A non-coding task is anything else: general knowledge, creative writing,
translation, personal advice, math derivations with no code, etc."""

_CLARIFICATION_SYSTEM = """\
You are the intake phase of a coding agent. Your goal is to gather exactly
enough information to write and test code — no more, no less.

Given the conversation so far, decide if there is ONE critical piece of
information still missing that would materially affect the implementation.

If you need more information, ask EXACTLY ONE short, specific question.
Format your response as:
  QUESTION: <your question here>

If you have enough information, respond with exactly:
  SUFFICIENT

Rules:
- Do NOT ask about things that can be reasonably assumed (e.g., language
  already mentioned, standard error handling, obvious defaults).
- Do NOT ask about edge cases that can be handled with sensible defaults.
- Only ask about genuine ambiguities that would cause fundamentally different
  implementations.
- Never ask more than one question at a time."""

_SUFFICIENCY_SYSTEM = """\
You are reviewing whether a coding task description is complete enough to
begin implementation. Answer YES if all of the following are known or can
be reasonably assumed:
  - What the code should do (core behavior)
  - The programming language (or a safe default exists)
  - The primary inputs and outputs
  - Any hard constraints (performance, format, etc.)

Answer NO if any of these are genuinely unclear.

Reply with exactly one word: YES or NO."""

_METRICS_GENERATION_SYSTEM = """\
You are defining what success looks like for a coding task.

Given the conversation, generate a concise list of success metrics — concrete,
testable criteria that would confirm the implementation is correct and complete.

Good metrics are specific and verifiable, for example:
  - "Returns the correct sum for positive, negative, and zero inputs"
  - "Raises ValueError when given an empty list"
  - "Processes a 1 GB file in under 10 seconds"
  - "All public methods have passing unit tests"

Bad metrics are vague:
  - "Works correctly" (not testable)
  - "Is fast" (not measurable)

Output ONLY a numbered list. No preamble, no explanation. Example format:
1. <metric>
2. <metric>
3. <metric>"""

_EXTRACTION_SYSTEM = """\
You are extracting a structured task specification from a conversation.

Produce a JSON object with exactly these fields:
{
  "refined_description": "<one or two sentences: the canonical problem statement>",
  "language": "<programming language, lowercase, e.g. python>",
  "task_type": "<one of: function | class | script | api_endpoint | data_pipeline | other_coding>",
  "requirements": ["<functional requirement sentence>", ...],
  "constraints": ["<non-functional constraint sentence>", ...],
  "dependencies": ["<library name>", ...],
  "examples": [{"input": "<value>", "output": "<value>"}, ...],
  "success_metrics": ["<confirmed success metric>", ...]
}

Rules:
- requirements: functional requirements only (what it must do)
- constraints: non-functional (performance, size, format)
- dependencies: only if explicitly mentioned by the user; empty list otherwise
- examples: only if input/output pairs were given; empty list otherwise
- success_metrics: use the user-confirmed metrics from the conversation
- Output ONLY the JSON object. No markdown fences. No explanation."""

_CLARIFICATION_TRANSITION_SYSTEM = """\
You are the friendly face of a coding agent. The user just submitted a coding
request. Write a single short sentence (max 15 words) that:
  - Acknowledges their request warmly
  - Sets the expectation that you'll ask a clarifying question

Do NOT ask the question. Do NOT use markdown. Output only the sentence."""

_METRICS_TRANSITION_SYSTEM = """\
You are the friendly face of a coding agent. You have just finished clarifying
a coding task and are about to propose success metrics. Write a single short
sentence (max 15 words) that:
  - Signals you have a good grasp of the task
  - Introduces the upcoming metrics naturally

Do NOT list the metrics. Do NOT use markdown. Output only the sentence."""

_METRICS_REVISION_SYSTEM = """\
You are updating a list of success metrics based on user feedback.

You will be given the current metrics and the user's requested changes.
Output ONLY the revised numbered list of metrics — no preamble, no explanation.
Apply the user's changes exactly: add, remove, or reword items as requested.

Format:
1. <metric>
2. <metric>
..."""

_REFUSAL_MESSAGE = (
    "I can only help with coding and programming tasks. "
    "Please describe a software problem — for example: writing a function, "
    "debugging code, designing a class, or implementing an algorithm."
)


# ── Heuristics ────────────────────────────────────────────────────────────────

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


# ── IntakeAgent ───────────────────────────────────────────────────────────────

class IntakeAgent:
    """
    Handles the initial intake phase of the iterative coding agent.

    Lifecycle (called once per user submission):
        spec = intake_agent.run(user_prompt)
        if spec is None:
            # non-coding request; refusal already streamed to UI
            return
        # pass spec to next pipeline stage
    """

    MAX_QUESTIONS: int = 4

    def __init__(self, bridge: BridgeProtocol) -> None:
        self._bridge = bridge
        self._history: list[dict[str, str]] = []

    def run(self, user_prompt: str) -> TaskSpec | None:
        """
        Entry point. Blocking — runs entirely in the worker thread.
        Returns TaskSpec on success, None if the request was refused.
        """
        self._history = []

        # Phase 1: pre-check
        self._bridge.post_activity("status", "checking request type...")
        if not self._is_coding_task(user_prompt):
            self._bridge.post_token(_REFUSAL_MESSAGE)
            return None

        # Seed conversation history with the user's prompt
        self._history.append({"role": "user", "content": user_prompt})

        # Phase 2: clarification loop
        self._bridge.post_activity("status", "analyzing request...")
        self._bridge.post_message(self._generate_transition(
            user_prompt, _CLARIFICATION_TRANSITION_SYSTEM
        ))
        turns = self._run_clarification_loop()

        # Phase 3: propose and confirm success metrics
        self._bridge.post_activity("status", "defining success metrics...")
        self._bridge.post_message(self._generate_transition(
            user_prompt, _METRICS_TRANSITION_SYSTEM
        ))
        self._confirm_success_metrics()

        # Phase 4: extract structured spec
        self._bridge.post_activity("status", "finalizing task specification...")
        spec = self._extract_task_spec(user_prompt, clarification_turns=turns)

        self._bridge.post_activity(
            "status",
            f"intake complete — {spec.task_type} / {spec.language} "
            f"/ {turns} clarification turn(s)",
        )
        return spec

    # ── Transition helper ─────────────────────────────────────────────────────

    def _generate_transition(self, user_prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        return self._bridge.call_llm(messages).strip()

    # ── Phase 1: pre-check ────────────────────────────────────────────────────

    def _is_coding_task(self, prompt: str) -> bool:
        result = self._heuristic_check(prompt)
        if result is not None:
            return result
        # Ambiguous — escalate to LLM
        self._bridge.post_activity("status", "classifying request...")
        return self._llm_precheck(prompt)

    def _heuristic_check(self, prompt: str) -> bool | None:
        lower = prompt.lower()
        coding_hits = sum(1 for sig in _CODING_SIGNALS if sig in lower)
        non_coding_hits = sum(1 for sig in _NON_CODING_SIGNALS if sig in lower)

        if coding_hits >= 2 and non_coding_hits == 0:
            return True
        if non_coding_hits >= 1 and coding_hits == 0:
            return False
        return None  # ambiguous

    def _llm_precheck(self, prompt: str) -> bool:
        messages = [
            {"role": "system", "content": _PRECHECK_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = self._bridge.call_llm(messages).strip()
        return response.upper().startswith("Y")

    # ── Phase 2: clarification loop ───────────────────────────────────────────

    def _run_clarification_loop(self) -> int:
        turns = 0
        for _ in range(self.MAX_QUESTIONS):
            question = self._generate_next_question(turns)
            if question is None:
                break  # LLM says it has enough information

            answer = self._bridge.ask_user(question)
            turns += 1

            # Record Q&A in history so subsequent calls have full context
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

        # Fallback: if the response looks like a question, use it directly
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
                    f"{m['role'].upper()}: {m['content']}" for m in self._history
                ),
            },
        ]
        response = self._bridge.call_llm(messages).strip()
        return response.upper().startswith("Y")

    # ── Phase 3: success metrics ──────────────────────────────────────────────

    def _confirm_success_metrics(self) -> None:
        """
        Generate proposed success metrics, then loop until the user confirms
        they are correct. Revisions are applied by the LLM and re-presented.
        The final confirmed metrics are recorded in history for extraction.
        """
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        )
        messages = [
            {"role": "system", "content": _METRICS_GENERATION_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{conversation_text}"},
        ]
        metrics = self._bridge.call_llm(messages).strip()

        while True:
            answer = self._bridge.ask_user(
                f"Here are the proposed success metrics for this task:\n\n"
                f"{metrics}\n\n"
                "Do these look correct? (yes / no — or describe what to change)"
            )

            if answer.strip().lower().startswith("y"):
                break

            feedback = answer.strip()
            if feedback.lower() in {"no", "n", "nope", "nah"}:
                feedback = self._bridge.ask_user(
                    "What would you like to add, remove, or change?"
                )

            # Revise the metrics list based on feedback
            self._bridge.post_activity("status", "revising success metrics...")
            revision_messages = [
                {"role": "system", "content": _METRICS_REVISION_SYSTEM},
                {
                    "role": "user",
                    "content": f"Current metrics:\n{metrics}\n\nRequested changes: {feedback}",
                },
            ]
            metrics = self._bridge.call_llm(revision_messages).strip()

        # Record the confirmed metrics in history so extraction picks them up
        confirmed_exchange = (
            f"Here are the confirmed success metrics:\n{metrics}"
        )
        self._history.append({"role": "assistant", "content": confirmed_exchange})
        self._history.append({"role": "user", "content": "Yes, those metrics look correct."})

    # ── Phase 4: extraction ───────────────────────────────────────────────────

    def _extract_task_spec(
        self, original_prompt: str, clarification_turns: int
    ) -> TaskSpec:
        conversation_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self._history
        )
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": f"Conversation:\n{conversation_text}"},
        ]
        raw = self._bridge.call_llm(messages).strip()

        # Strip markdown fences if present
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
                success_metrics=data.get("success_metrics", []),
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
                clarification_turns=clarification_turns,
            )
