"""LangChain agent logic for the financial advisory agent."""

import os

try:
    from dotenv import load_dotenv
    from langchain_anthropic import ChatAnthropic
except ImportError as err:
    # The lab handout's `pip install openai langchain streamlit pandas` misses
    # langchain-anthropic and python-dotenv, so point at the real install line.
    raise ImportError(
        f"Missing dependency '{err.name}'. Install this project's requirements:\n"
        f"    pip install -r requirements.txt"
    ) from err

from advisor_prompt import advisor_template, followup_template
from mock_data import format_expenses_for_prompt, get_user_financial_data

load_dotenv()

# Model config is shared with the agents/ package so the five implementations
# cannot drift apart. Claude Opus 5 rejects temperature/top_p/top_k, so depth
# is steered with adaptive thinking + effort instead of a sampling temperature.
from agents.settings import (  # noqa: E402  (after load_dotenv by design)
    EFFORT,
    MAX_TOKENS,
    MODEL,
    MissingAPIKeyError,
    require_api_key as _require_api_key,
)


def build_llm(streaming=False):
    """Create the Claude chat model used by every call in this module."""
    _require_api_key()
    return ChatAnthropic(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        streaming=streaming,
    )


def _extract_text(content):
    """Pull plain text out of a response.

    With adaptive thinking on, `content` is a list of blocks (thinking, text,
    ...) rather than a bare string, so pick out the text blocks.
    """
    if isinstance(content, str):
        return content

    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _profile_variables(user_data):
    """Map a profile onto the prompt template's input variables."""
    return {
        "income": f"{user_data['income']:,}",
        "expenses": format_expenses_for_prompt(user_data),
        "goals": ", ".join(user_data["financial_goals"]),
        "risk": user_data["risk_tolerance"],
    }


def _resolve(user_id, user_data):
    return user_data if user_data is not None else get_user_financial_data(user_id)


def _format_history(history, limit=6):
    """Render recent chat turns for the follow-up prompt."""
    if not history:
        return "(no prior questions)"

    return "\n\n".join(
        f"{'User' if role == 'user' else 'Advisor'}: {text}"
        for role, text in history[-limit:]
    )


def run_financial_advisor(user_id="user123", user_data=None):
    """Generate the full financial plan for a user. Returns markdown text."""
    user_data = _resolve(user_id, user_data)
    messages = advisor_template.format_messages(**_profile_variables(user_data))

    response = build_llm().invoke(messages)
    return _extract_text(response.content)


def stream_financial_advisor(user_id="user123", user_data=None):
    """Same as run_financial_advisor, but yields text chunks as they arrive."""
    user_data = _resolve(user_id, user_data)
    messages = advisor_template.format_messages(**_profile_variables(user_data))

    for chunk in build_llm(streaming=True).stream(messages):
        text = _extract_text(chunk.content)
        if text:
            yield text


def answer_followup(question, history=None, user_id="user123", user_data=None):
    """Answer a follow-up question about an already-delivered plan."""
    user_data = _resolve(user_id, user_data)
    messages = followup_template.format_messages(
        question=question,
        history=_format_history(history),
        **_profile_variables(user_data),
    )

    response = build_llm().invoke(messages)
    return _extract_text(response.content)


def stream_followup(question, history=None, user_id="user123", user_data=None):
    """Same as answer_followup, but yields text chunks as they arrive."""
    user_data = _resolve(user_id, user_data)
    messages = followup_template.format_messages(
        question=question,
        history=_format_history(history),
        **_profile_variables(user_data),
    )

    for chunk in build_llm(streaming=True).stream(messages):
        text = _extract_text(chunk.content)
        if text:
            yield text


if __name__ == "__main__":
    import sys

    user = sys.argv[1] if len(sys.argv) > 1 else "user123"
    try:
        print(run_financial_advisor(user_id=user))
    except MissingAPIKeyError as err:
        sys.exit(f"error: {err}")
