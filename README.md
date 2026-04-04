# IterativeCodingAgent

An iterative coding agent that integrates a lightweight coding model with a sandboxed shell environment. The agent takes a coding request, clarifies requirements, defines success metrics, generates tests, writes code, runs the tests, and iterates until the code passes.

![Architecture](https://github.com/user-attachments/assets/e6322086-bd0d-4e92-9064-f89a5dcfb912)

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Clone the repo
git clone <repo-url>
cd IterativeCodingAgent

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### LLM Provider

You need at least one of:

**Ollama (local)**
- Install from [ollama.com](https://ollama.com) and pull a model, e.g.:
  ```bash
  ollama pull qwen2.5-coder:7b
  ```
- Thinking models (e.g. `qwq`, `deepseek-r1`) will display internal reasoning in the agent pane
- Ollama must be running before you launch the app (`ollama serve`)

**TAMU LLM (remote)**
- Requires a TAMU API key
- Endpoint: `https://chat-api.tamu.ai/openai/chat/completions`
- Models that return `reasoning_content` or inline `<think>` tags will show thinking traces

---

## Run

```bash
python3 -m cli.app
```

On the setup screen, pick a provider, enter credentials (API key for TAMU, host URL for Ollama), fetch available models, then connect.

---

## Debugging

All debug output is written to `bridge-debug.log` in the project root (Textual owns the terminal while running, so `print` is not visible). Tail it in a second terminal:

```bash
tail -f bridge-debug.log
```

---

## What's Implemented

### CLI (`cli/`)

A terminal UI built with [Textual](https://github.com/Textualize/textual), split into a 60/40 layout:

- **Left pane (conversation)** — full chat history with per-message widgets. Agent messages, user messages, and agent questions are distinct styled bubbles. Questions that include a file preview embed a collapsible code block inline (expand/collapse with a click), similar to Claude Code.
- **Right pane (agent activity)** — timestamped log of status updates, thinking traces (rendered as Markdown), task-ready summaries, and errors.
- **Setup screen** — LLM provider configuration (Ollama or TAMU LLM), model selection, and connection management.

### AGENT (`agent/`)

#### Intake Phase (`agent/intake.py`)

Runs before any code is written. Produces a structured `TaskSpec` consumed by downstream phases.

**Flow:**

```
User prompt
  → pre-check (coding task? — heuristic + LLM classifier)
  → conversational acknowledgement  ← LLM-generated
  → clarification loop (targeted Q&A, up to 4 rounds)
  → conversational transition        ← LLM-generated
  → propose success metrics → user confirms / revises in a loop
  → extract TaskSpec (language, type, requirements, constraints, metrics, ...)
  → display TaskSpec summary in agent pane
  → hand off to test generation
```

Key behaviours:
- **Clarification loop** — the agent asks at most 4 targeted questions, stopping early when it has enough context.
- **Metrics confirmation loop** — proposed metrics are shown to the user; feedback triggers an LLM revision pass and re-presentation until the user approves.
- **Conversational transitions** — the agent posts a short LLM-generated message before each major phase change so the interaction feels natural rather than jumping straight to a question.

#### Test Generation Phase (`agent/test_generator.py`)

Receives the finalized `TaskSpec` and writes a test suite using TDD principles.

**Flow:**

```
TaskSpec received
  → conversational acknowledgement   ← LLM-generated
  → LLM writes test suite covering all success metrics and requirements
  → conversational handoff message   ← LLM-generated (mentions line count + metric count)
  → show proposed test file in collapsible "agent asks" bubble
  → user approval loop — loops until user says yes
  → write test file to workspace/
```

Key behaviours:
- **Retry loop** — if the LLM fails to produce a valid code block, it is prompted to fix it (up to 2 retries).
- **User approval gate** — the generated file is shown inline before any disk write; the agent waits until the user explicitly approves.
- **Thinking traces** — internal LLM reasoning is extracted from dedicated fields (`message.thinking`, `reasoning_content`) or inline `<think>` tags and displayed in the agent pane.

---

## Planned (not yet implemented)

- Code generation from `TaskSpec` + test suite
- Sandboxed test execution
- Analyze results and propose fixes (iterative loop)

---
