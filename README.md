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
- Ollama must be running before you launch the app (`ollama serve`)

**TAMU LLM (remote)**
- Requires a TAMU API key
- Endpoint: `https://chat-api.tamu.ai/openai/chat/completions`

---

## Run

```bash
python3 -m cli.app
```

On the setup screen, pick a provider, enter credentials (API key for TAMU, host URL for Ollama), fetch available models, then connect.

---

## What's Implemented

### CLI (`cli/`)

A terminal UI built with [Textual](https://github.com/Textualize/textual). It provides:
- A setup screen for LLM provider configuration
- A main interface for entering coding requests and viewing the agent's iterative process

### AGENT (`agent/`)

#### Intake Phase 

The intake phase runs before any code is written. It handles the initial user interaction and produces a structured `TaskSpec` that downstream phases (test generation, coding, etc.) will consume.

**Intake flow:**

```
User prompt
  → pre-check (coding task?)
  → clarification loop (targeted Q&A, max 4 rounds)
  → propose success metrics → user confirms / revises
  → extract TaskSpec (language, type, requirements, constraints, metrics, ...)
  → hand off to next pipeline stage
```

---

## Planned (not yet implemented)

- Test generation from `TaskSpec`
- Code generation
- Sandboxed test execution
- Analyze results and propose fixes (iterative loop)

---
