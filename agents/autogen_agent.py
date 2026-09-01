"""AutoGen multi-agent conversation on Claude.

The widely-copied version of this template no longer runs on any current
release:

1. `from autogen import AssistantAgent, UserProxyAgent` - that namespace is
   gone. `pyautogen` is now a shim over `autogen-agentchat`, whose agents live
   in `autogen_agentchat.agents` and are async.
2. `llm_config={"config_list": [{"api_type": "anthropic", ...}]}` was replaced
   by model client objects - here `AnthropicChatCompletionClient`.
3. `temperature` is rejected by Opus 5, so it is left unset.
4. AutoGen auto-detects unknown model ids as the claude-3-5-sonnet family with
   `function_calling: False`, which silently disables tool use. Passing an
   explicit `model_info` is what keeps the tools working.

Requires the optional extras:  pip install -r requirements-agents.txt
"""

import asyncio

try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import (
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core.models import ModelFamily, ModelInfo
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient
except ImportError as err:
    raise ImportError(
        f"Missing dependency '{err.name}'. AutoGen is an optional extra:\n"
        f"    pip install -r requirements-agents.txt"
    ) from err

from agents.settings import MAX_TOKENS, MODEL, require_api_key
from agents.tools import get_financial_profile, project_savings, time_to_goal

# Without this, AutoGen guesses the model family and turns tool calling off.
CLAUDE_MODEL_INFO = ModelInfo(
    vision=True,
    function_calling=True,
    json_output=True,
    family=ModelFamily.UNKNOWN,  # no Opus 5 entry in this enum yet
    structured_output=True,
    multiple_system_messages=False,
)

MAX_TURNS = 12


# --- tools ---------------------------------------------------------------------
# AutoGen reads the signature and docstring to build the schema, so the type
# hints and the wording here are the tool contract.

def lookup_profile(user_id: str) -> str:
    """Look up a user's income, expenses, goals, risk tolerance, and derived
    monthly figures (take-home pay, surplus, savings rate)."""
    return str(get_financial_profile(user_id))


def project_growth(
    monthly_contribution: float, years: float, annual_return_pct: float = 6.0
) -> str:
    """Project what a monthly contribution grows to over a number of years,
    compounded monthly."""
    return str(project_savings(monthly_contribution, years, annual_return_pct))


def months_to_goal(
    target_amount: float, monthly_contribution: float, annual_return_pct: float = 6.0
) -> str:
    """Compute how many months of a monthly contribution are needed to reach a
    target amount, including growth."""
    return str(time_to_goal(target_amount, monthly_contribution, annual_return_pct))


def build_model_client():
    """Claude, configured for AutoGen. No temperature - Opus 5 rejects it."""
    api_key = require_api_key()
    return AnthropicChatCompletionClient(
        model=MODEL,
        api_key=api_key,
        max_tokens=MAX_TOKENS,
        model_info=CLAUDE_MODEL_INFO,
    )


def build_team(model_client=None):
    """An analyst and a reviewer talking until the plan is signed off."""
    model_client = model_client or build_model_client()

    analyst = AssistantAgent(
        name="Financial_Analyst",
        model_client=model_client,
        tools=[lookup_profile, project_growth, months_to_goal],
        system_message=(
            "You are a financial analyst. Call lookup_profile before giving any "
            "advice, and use the projection tools for forward-looking numbers - "
            "never estimate compound growth yourself. Produce a plan with three "
            "sections: where the user stands, a monthly savings target, and an "
            "investment strategy matched to their risk tolerance. Revise it when "
            "the reviewer objects."
        ),
    )

    reviewer = AssistantAgent(
        name="Compliance_Reviewer",
        model_client=model_client,
        system_message=(
            "You review draft financial plans. Check that every figure traces "
            "to the profile, the arithmetic is shown, the savings target is a "
            "specific dollar amount, and the allocation matches the stated risk "
            "tolerance. List concrete problems if any. When the plan satisfies "
            "all four, reply with the single word APPROVED."
        ),
    )

    # Stop on sign-off, with a message cap so a disagreement cannot run forever.
    termination = TextMentionTermination("APPROVED") | MaxMessageTermination(MAX_TURNS)

    return RoundRobinGroupChat([analyst, reviewer], termination_condition=termination)


async def run_autogen_agent_async(user_id="user123", model_client=None):
    """Run the conversation and return the transcript's final plan."""
    client = model_client or build_model_client()
    team = build_team(client)

    task = (
        f"Build a financial plan for user '{user_id}'. "
        f"Look up their profile first."
    )
    result = await team.run(task=task)

    try:
        # The last substantive message before sign-off is the plan itself.
        for message in reversed(result.messages):
            text = getattr(message, "content", "")
            if isinstance(text, str) and text.strip() != "APPROVED":
                return text
        return ""
    finally:
        await client.close()


def run_autogen_agent(user_id="user123"):
    """Synchronous wrapper - AutoGen's current API is async."""
    return asyncio.run(run_autogen_agent_async(user_id))


if __name__ == "__main__":
    import sys

    from agents.settings import MissingAPIKeyError

    try:
        print(run_autogen_agent(sys.argv[1] if len(sys.argv) > 1 else "user123"))
    except MissingAPIKeyError as err:
        sys.exit(f"error: {err}")
