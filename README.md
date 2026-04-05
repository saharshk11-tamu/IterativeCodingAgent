# IterativeCodingAgent

An iterative coding agent with a Textual terminal UI. The current implementation takes a coding request, clarifies requirements, defines success metrics, generates tests, and saves the proposed test file after user approval.

![Architecture](https://github.com/user-attachments/assets/e6322086-bd0d-4e92-9064-f89a5dcfb912)

---

## Setup

Requirements: Python 3.11+

```bash
git clone <repo-url>
cd IterativeCodingAgent
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Hosted Providers

This milestone supports:

- OpenAI
- Anthropic
- Gemini

You need an API key for at least one provider. The setup screen can fetch model IDs for supported providers, and manual model entry is always available.

---

## Run

```bash
python3 -m cli.app
```

On the setup screen, pick a provider, enter an API key, optionally fetch models, enter a model ID, then connect.

---

## Debugging

All debug output is written to `bridge-debug.log` in the project root. Because Textual owns the terminal while running, normal `print` output is not visible.

```bash
tail -f bridge-debug.log
```

---

## What's Implemented

### CLI (`cli/`)

A Textual UI with a 60/40 layout:

- Left pane: full conversation history with separate widgets for user messages, agent messages, and agent questions.
- Right pane: timestamped status updates, task summaries, and errors.
- Setup screen: hosted LLM provider configuration for OpenAI, Anthropic, and Gemini, with optional model fetching and manual model entry.

### Agent (`agent/`)

#### Intake Phase (`agent/intake.py`)

Produces a structured `TaskSpec` before any code is written.

Flow:

```text
User prompt
  -> pre-check (coding task? - heuristic + LLM classifier)
  -> conversational acknowledgement
  -> clarification loop (targeted Q&A, up to 4 rounds)
  -> conversational transition
  -> propose success metrics -> user confirms / revises in a loop
  -> extract TaskSpec (language, type, requirements, constraints, metrics, ...)
  -> display TaskSpec summary in agent pane
  -> hand off to test generation
```

Key behaviors:

- Clarification loop capped at 4 targeted questions.
- Metrics confirmation loop until user approval.
- Short conversational transitions before major phase changes.

#### Test Generation Phase (`agent/test_generator.py`)

Receives a finalized `TaskSpec` and writes a test suite using TDD-style prompting.

Flow:

```text
TaskSpec received
  -> conversational acknowledgement
  -> LLM writes test suite covering requirements and success metrics
  -> conversational handoff message
  -> show proposed test file in collapsible "agent asks" bubble
  -> user approval loop
  -> write test file to workspace/
```

Key behaviors:

- Retry loop if the LLM fails to return a usable code block.
- User approval gate before any file write.

### Hosted Provider Layer (`llm/`)

The app now uses a shared provider abstraction for:

- OpenAI Chat Completions
- Anthropic Messages
- Gemini `generateContent`

The bridge and agent modules use one normalized `call_llm(messages)` path regardless of provider.

---

## Planned

- Code generation from `TaskSpec` plus the approved test suite
- Sandboxed test execution
- Analyze results and iteratively propose or apply fixes
