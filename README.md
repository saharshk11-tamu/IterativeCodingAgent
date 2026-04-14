# IterativeCodingAgent

An iterative coding agent with a Textual terminal UI. The current implementation can:

- connect to hosted LLM providers and local Ollama models
- clarify a coding task
- confirm structured executable metrics and targets
- generate evaluator artifacts under `workspace/.agent/`
- run an automated Python-first improvement loop inside `workspace/`
- execute evaluation and helper Python commands through `uv`

![Architecture](https://github.com/user-attachments/assets/e6322086-bd0d-4e92-9064-f89a5dcfb912)

---

## Setup

Requirements: Python 3.12+, `uv`

```bash
git clone <repo-url>
cd IterativeCodingAgent
uv sync
```

### Providers

Supported providers:

- OpenAI
- Anthropic
- Gemini
- Ollama

You need an API key for hosted providers. For Ollama, the app uses the local OpenAI-compatible API at `http://localhost:11434/v1` and does not prompt for an API key.

The setup screen can fetch model IDs for supported providers. For Ollama, model discovery comes from the local Ollama instance, and manual model entry is always available.

---

## Run

```bash
uv run python -m cli.app
```

On the setup screen, pick a provider, enter an API key if needed, optionally fetch models, enter a model ID, then connect. For Ollama, use `F5` to query your local models or enter one manually.

---

## Workflow

The current workflow is:

```text
User prompt
  -> coding-task pre-check
  -> clarification loop
  -> structured metric and target confirmation
  -> TaskSpec extraction
  -> workspace grounding
  -> evaluator generation under workspace/.agent/
  -> baseline evaluator run
  -> automated implementation loop in workspace/
  -> stop on target hit, plateau, iteration budget, or time limit
  -> final metric summary and run report
```

### Workspace Runtime

The automation phase uses a fixed tool API rather than giving the model direct shell access. The agent may modify only `workspace/`.

Python execution is mediated by `uv`:

- evaluator runs
- Python test runs
- Python script runs
- transient package installs via `uv run --with ...`

Transient package installs do not modify `pyproject.toml` or `uv.lock`. Run artifacts are written under `workspace/.agent/`.

---

## Debugging

All debug output is written to `bridge-debug.log` in the project root. Because Textual owns the terminal while running, normal `print` output is not visible.

```bash
tail -f bridge-debug.log
```

---

## What's Implemented

### CLI (`cli/`)

- 60/40 Textual layout for conversation and agent activity
- provider setup for OpenAI, Anthropic, Gemini, and Ollama
- activity log entries for status, tool calls, tool results, and metrics

### Agent (`agent/`)

- intake with clarification and structured metric confirmation
- evaluator/result typing and metric comparison logic
- workspace-only runtime tools
- autonomous improvement loop with baseline, iteration control, plateau detection, and final reporting

### Hosted Provider Layer (`llm/`)

- OpenAI Chat Completions
- Anthropic Messages
- Gemini `generateContent`
- Ollama via the OpenAI-compatible local API

All provider calls are normalized behind a shared `call_llm(messages)` path.

---

## Current Constraints

- Python-first execution only
- workspace-only edits during automated runs
- no direct shell access for the agent
- network access during task execution is limited to transient package installation
