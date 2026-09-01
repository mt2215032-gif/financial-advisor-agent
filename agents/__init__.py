"""Four agent implementations of the same financial advisor.

Each module is self-contained and shares the model config in `settings.py` and
the deterministic money math in `tools.py`:

    native_agent   Anthropic SDK tool-use loop, no framework
    graph_agent    LangGraph state machine with a draft/review cycle
    crew_agent     CrewAI role-playing team (optional extra)
    autogen_agent  AutoGen multi-agent conversation (optional extra)

The framework modules are not imported here - crewai and autogen are optional,
and importing this package must not require them.
"""

from agents.settings import EFFORT, MAX_TOKENS, MODEL, MissingAPIKeyError

__all__ = ["MODEL", "EFFORT", "MAX_TOKENS", "MissingAPIKeyError"]
