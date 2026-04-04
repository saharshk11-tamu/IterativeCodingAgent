"""
agent/test_generator.py — Test generation phase.

Responsibilities:
  1. Take a finalized TaskSpec.
  2. Prompt the LLM to write a comprehensive test suite based on the spec's metrics.
  3. Extract the raw test code.
  4. Save the test code to a workspace directory for later execution.
"""

import json
import re
import os
from agent.task_spec import TaskSpec
from agent.intake import BridgeProtocol

_GENERATION_START_SYSTEM = """\
You are the friendly face of a coding agent moving into the test generation phase.
The user's task spec is ready. Write a single short sentence (max 15 words) that:
  - Signals you're now going to write the test suite
  - Optionally references the language or task type

Do NOT write any tests. Do NOT use markdown. Output only the sentence."""

_GENERATION_DONE_SYSTEM = """\
You are the friendly face of a coding agent that just finished writing a test suite.
Write a single short sentence (max 20 words) that:
  - Tells the user you've written the tests
  - Mentions the number of lines and invites them to review

Do NOT use markdown. Output only the sentence.

You will be given: language, line count, number of success metrics covered."""

_TEST_SYSTEM_PROMPT = """\
You are an expert software tester acting as the Test Generation module for an autonomous coding agent. 
Your goal is to write a comprehensive, robust test suite for a task before the implementation is written (Test-Driven Development).

You will be given a JSON specification of the task, including requirements, constraints, and success metrics.

Rules:
1. Write tests in the language specified in the JSON. If Python, use `pytest` (preferred) or the built-in `unittest` module.
2. Write ONLY the test code. Do NOT write the implementation of the target code. Assume the target function/class will be available in the environment (e.g., assume it will be imported or defined directly above).
3. Ensure every single item in the `success_metrics` and `requirements` lists is explicitly covered by a test.
4. Output the code inside a standard markdown code block (e.g., ```python ... ```). Do not include any other conversational text, explanations, or preamble.
"""

class TestGenerator:
    def __init__(self, bridge: BridgeProtocol):
        self._bridge = bridge

    def _generate_transition(self, system: str, user_content: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return self._bridge.call_llm(messages).strip()

    def generate_and_save(self, spec: TaskSpec, workspace_dir: str = "./workspace") -> str:
        """
        Generates the tests, shows them to the user, then waits for approval
        before writing any files. Loops until the user agrees.
        """
        self._bridge.post_activity("status", "generating test suite...")
        self._bridge.post_message(self._generate_transition(
            _GENERATION_START_SYSTEM,
            f"Language: {spec.language}, task type: {spec.task_type}, "
            f"description: {spec.refined_description}",
        ))

        test_code = self.generate(spec)

        ext = "py" if spec.language.lower() == "python" else "txt"
        file_path = os.path.join(workspace_dir, f"test_solution.{ext}")
        line_count = len(test_code.splitlines())

        self._bridge.post_message(self._generate_transition(
            _GENERATION_DONE_SYSTEM,
            f"Language: {spec.language}, lines: {line_count}, "
            f"success metrics covered: {len(spec.success_metrics)}",
        ))

        # Loop until the user approves saving — show file contents each time
        lang = spec.language.lower()
        while True:
            answer = self._bridge.ask_user(
                f"**Proposed test file** — `{file_path}` ({line_count} lines):\n\n"
                f"```{lang}\n{test_code}\n```\n\n"
                f"Save this file? (yes/no)"
            )
            if answer.strip().lower().startswith("y"):
                break
            self._bridge.post_activity("status", "save declined — waiting for approval...")

        # Create workspace if it doesn't exist
        os.makedirs(workspace_dir, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(test_code)

        self._bridge.post_activity("status", f"test generation complete — {line_count} lines saved to {file_path}")

        return test_code

    def generate(self, spec: TaskSpec, max_retries: int = 2) -> str:
        """
        Takes the TaskSpec from the intake phase and returns a string of test code.
        Includes a retry loop in case the LLM fails to output valid code blocks.
        """
        # We format the relevant parts of the spec into JSON for strict LLM parsing
        spec_dict = {
            "refined_description": spec.refined_description,
            "language": spec.language,
            "requirements": spec.requirements,
            "constraints": spec.constraints,
            "examples": spec.examples,
            "success_metrics": spec.success_metrics
        }

        messages = [
            {"role": "system", "content": _TEST_SYSTEM_PROMPT},
            {"role": "user", "content": f"Task Specification:\n```json\n{json.dumps(spec_dict, indent=2)}\n```"}
        ]

        self._bridge.post_activity("thinking", "Drafting tests based on TaskSpec metrics...")
        
        for attempt in range(max_retries):
            # Call the LLM
            raw_response = self._bridge.call_llm(messages)

            # Extract the actual code from the markdown fences
            test_code = self._extract_code(raw_response)

            # Simple validation: Check if it actually extracted something that looks like code
            if test_code and ("import " in test_code or "def test_" in test_code or "assert" in test_code):
                return test_code
                
            self._bridge.post_activity("error", f"Attempt {attempt + 1} failed to generate valid code. Retrying...")
            
            # Add the failed response to the prompt to tell it to fix it
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": "You forgot to output the code in a markdown block, or the code was invalid. Please output ONLY the raw test code inside a markdown block."})
            
        # Fallback if it fails completely after retries
        error_msg = "# Error: Failed to generate valid test code."
        self._bridge.post_activity("error", "Test generation failed after maximum retries.")
        return error_msg

    def _extract_code(self, response: str) -> str:
        """Helper to rip the code out of standard Markdown fences."""
        match = re.search(r"```(?:\w+)?\s*\n(.*?)```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response.strip()
