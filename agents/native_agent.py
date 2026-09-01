"""Native Anthropic SDK tool-use loop - no framework, just the SDK.

Fixes four bugs in the common version of this template:

1. `response.content[0].text` assumes the first block is text. With thinking
   on it is a thinking block, so the answer is read by scanning for text
   blocks instead.
2. Handling only the first `tool_use` block silently drops the rest. Claude
   emits several in one turn (parallel tool use), and every result must come
   back in a single user message - splitting them teaches the model to stop
   parallelising.
3. `while True` never terminates if the model keeps calling tools.
4. Only `end_turn` and `tool_use` were handled; `max_tokens`, `refusal`, and
   `pause_turn` fell through and hung the loop.
"""

import anthropic

from agents.settings import (
    MAX_TOKENS,
    MODEL,
    OUTPUT_CONFIG,
    THINKING,
    require_api_key,
)
from agents.tools import TOOL_SCHEMAS, run_tool

SYSTEM_PROMPT = """\
You are a personal financial advisor.

Always call get_financial_profile before giving advice - never guess at a
user's numbers. Use project_savings and time_to_goal for any forward-looking
figure rather than estimating compound growth yourself.

Close with a three-section plan: where they stand, a monthly savings target,
and an investment strategy matched to their risk tolerance.\
"""

MAX_TURNS = 10


def extract_text(content):
    """Concatenate the text blocks, skipping thinking and tool_use blocks."""
    return "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )


def _tool_results_for(content):
    """Run every tool_use block in one assistant turn.

    Returns one tool_result per tool_use, in a single list - they must all go
    back in one user message.
    """
    results = []

    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue

        output, is_error = run_tool(block.name, dict(block.input))
        result = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": output,
        }
        if is_error:
            # Flagging the error lets the model fix its arguments and retry.
            result["is_error"] = True
        results.append(result)

    return results


def run_native_agent(prompt, max_turns=MAX_TURNS, client=None):
    """Run the tool loop until Claude produces a final answer.

    Returns the answer text. Raises RuntimeError if the model refuses or the
    loop exceeds max_turns.
    """
    require_api_key()
    client = client or anthropic.Anthropic()

    messages = [{"role": "user", "content": prompt}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            thinking=THINKING,
            output_config=OUTPUT_CONFIG,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            # Append the whole content, thinking blocks included - they must be
            # echoed back unchanged for the model to continue its reasoning.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": _tool_results_for(response.content)})
            continue

        if response.stop_reason == "pause_turn":
            # A server-side tool paused the turn; send it back to resume.
            messages.append({"role": "assistant", "content": response.content})
            continue

        if response.stop_reason == "refusal":
            # stop_details is populated only for refusals.
            details = response.stop_details
            category = getattr(details, "category", None) if details else None
            raise RuntimeError(f"Claude declined this request (category: {category})")

        if response.stop_reason == "max_tokens":
            # Return what we have, flagged, rather than pretending it is whole.
            return extract_text(response.content) + "\n\n[truncated: hit max_tokens]"

        return extract_text(response.content)

    raise RuntimeError(f"Tool loop did not finish within {max_turns} turns")


if __name__ == "__main__":
    import sys

    from agents.settings import MissingAPIKeyError

    question = " ".join(sys.argv[1:]) or (
        "Build a financial plan for user123. How long until they can put "
        "$60,000 down on a house?"
    )
    try:
        print(run_native_agent(question))
    except MissingAPIKeyError as err:
        sys.exit(f"error: {err}")
