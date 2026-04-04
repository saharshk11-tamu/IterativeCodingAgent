"""
cli/events.py — Custom Textual messages

Three categories:
  - User-initiated:  UserSubmitted, UserReplied
  - Agent streaming: AgentToken, AgentActivity, AgentDone
  - Agent-initiated: AgentQuestion  ← pauses the agent until UserReplied arrives
"""

from textual.message import Message

# ── User → Agent ──────────────────────────────────────────────────────────────


class UserSubmitted(Message):
    """User pressed Enter with a new top-level prompt."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class UserReplied(Message):
    """
    User answered an agent clarification question.
    ChatScreen catches this and puts the text onto bridge.reply_queue
    so the blocked agent thread can resume.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


# ── Agent → UI (streaming) ────────────────────────────────────────────────────


class AgentToken(Message):
    """A single streamed token from the LLM response."""

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token


class AgentMessage(Message):
    """
    A complete conversational message from the agent posted mid-flow
    (not streamed). Rendered as an agent bubble in the conversation pane.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class AgentActivity(Message):
    """
    A structured agent event shown in the right pane.

    kind: "thinking" | "tool_call" | "tool_result" | "status" | "error" | "done"
    """

    def __init__(self, kind: str, text: str) -> None:
        super().__init__()
        self.kind = kind
        self.text = text


class AgentDone(Message):
    """Agent finished its current run (including all clarification loops)."""

    pass


# ── Agent → UI (clarification) ────────────────────────────────────────────────


class AgentQuestion(Message):
    """
    Agent needs more context before continuing.

    Posting this:
      1. Shows the question as an agent bubble in the conversation pane
      2. Switches PromptInput into reply mode (different placeholder + submit behaviour)
      3. The agent thread is already blocked on reply_queue.get() by this point

    question: the text the agent wants to ask the user
    """

    def __init__(self, question: str) -> None:
        super().__init__()
        self.question = question
