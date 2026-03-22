# IterativeCodingAgent
An Iterative Coding Agent that integrates a lightweight coding model with a sandboxed shell environment

Agent Structure
![Untitled](https://github.com/user-attachments/assets/e6322086-bd0d-4e92-9064-f89a5dcfb912)

---

## CLI

A terminal UI for interacting with the agent, built with [Textual](https://github.com/Textualize/textual).

**What's implemented**
- Provider selection — Ollama (local) or TAMU LLM (remote, OpenAI-compatible)
- Model fetch — pulls available models from the selected provider on demand
- Split-pane chat — conversation on the left, agent activity log on the right
- Multiline input — `Enter` to send

**Setup**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Run**

```bash
python3 -m cli.app
```

On the setup screen, pick a provider, enter credentials, fetch models, then connect. Already tested TAMU LLM.
