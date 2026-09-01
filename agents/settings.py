"""Model configuration shared by every agent framework in this package.

One place to change the model, so the LangGraph / CrewAI / AutoGen / native
implementations cannot drift apart.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Claude Opus 5. Model IDs for current models carry no date suffix - appending
# one (claude-opus-5-20260101) is rejected as an unknown model.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

# Reasoning depth: low | medium | high | xhigh | max.
EFFORT = os.getenv("ANTHROPIC_EFFORT", "medium")

# Generous, because hitting the cap truncates mid-sentence and costs a retry.
# You are billed for tokens generated, not for the ceiling.
MAX_TOKENS = 16000

# Adaptive thinking. Opus 5 has no fixed thinking budget - `budget_tokens` is
# rejected - and it replaces the sampling temperature these templates used to
# set, which Opus 5 also rejects.
THINKING = {"type": "adaptive"}

OUTPUT_CONFIG = {"effort": EFFORT}


class MissingAPIKeyError(RuntimeError):
    """Raised when no Anthropic credential is available."""


def require_api_key():
    """Fail early and clearly rather than deep inside a framework call."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key from https://console.anthropic.com/settings/keys"
        )
    return os.environ["ANTHROPIC_API_KEY"]
