"""LangChain agent logic for the financial advisory agent."""

import os

try:
    from dotenv import load_dotenv
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

# Model config is shared with the agents/ package so the implementations cannot
# drift apart. `providers` picks Claude, OpenAI, or an open-source model from
# LLM_PROVIDER and sends only the parameters that provider accepts.
from agents.providers import (  # noqa: E402  (after load_dotenv by design)
    MissingCredentialError,
    build_chat_model,
    describe_active,
    get_provider,
    model_name_for,
)
from agents.settings import (  # noqa: E402
    EFFORT,
    MAX_TOKENS,
    MissingAPIKeyError,
)

# Kept for callers that just want "which model is this". Resolved through the
# provider layer so it follows LLM_PROVIDER rather than assuming Claude.
MODEL = model_name_for(get_provider())


def build_llm(streaming=False, provider=None):
    """Create the chat model for the active provider."""
    return build_chat_model(provider=provider, streaming=streaming)


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


def run_financial_advisor(user_id="user123", user_data=None, provider=None):
    """Generate the full financial plan for a user. Returns markdown text."""
    user_data = _resolve(user_id, user_data)
    messages = advisor_template.format_messages(**_profile_variables(user_data))

    response = build_llm(provider=provider).invoke(messages)
    return _extract_text(response.content)


def stream_financial_advisor(user_id="user123", user_data=None, provider=None):
    """Same as run_financial_advisor, but yields text chunks as they arrive."""
    user_data = _resolve(user_id, user_data)
    messages = advisor_template.format_messages(**_profile_variables(user_data))

    for chunk in build_llm(streaming=True, provider=provider).stream(messages):
        text = _extract_text(chunk.content)
        if text:
            yield text


def answer_followup(question, history=None, user_id="user123", user_data=None,
                    provider=None):
    """Answer a follow-up question about an already-delivered plan."""
    user_data = _resolve(user_id, user_data)
    messages = followup_template.format_messages(
        question=question,
        history=_format_history(history),
        **_profile_variables(user_data),
    )

    response = build_llm(provider=provider).invoke(messages)
    return _extract_text(response.content)


def stream_followup(question, history=None, user_id="user123", user_data=None,
                    provider=None):
    """Same as answer_followup, but yields text chunks as they arrive."""
    user_data = _resolve(user_id, user_data)
    messages = followup_template.format_messages(
        question=question,
        history=_format_history(history),
        **_profile_variables(user_data),
    )

    for chunk in build_llm(streaming=True, provider=provider).stream(messages):
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
