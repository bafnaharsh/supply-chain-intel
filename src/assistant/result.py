"""The one-turn result shared by the LLM runner, the offline router, the CLI and the API.

Kept free of ADK imports so offline mode never loads google.adk.
"""

from dataclasses import dataclass, field


@dataclass
class AnswerResult:
    text: str
    tools_called: list[str] = field(default_factory=list)
    llm_calls: int = 0
    usage: dict = field(default_factory=dict)
    guardrail: str | None = None
