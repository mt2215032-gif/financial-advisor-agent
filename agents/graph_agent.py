"""LangGraph state machine: draft a plan, review it, revise if needed.

The single-node graph these templates usually show is just one API call with
extra ceremony. A state machine earns its keep when there is a *cycle*, so
this one loops draft -> review -> draft until the reviewer approves or the
revision budget runs out.

    load_profile -> draft_plan -> review_plan --(revise)--> draft_plan
                                       |
                                    (approve)
                                       v
                                      END
"""

import operator
from typing import Annotated

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.settings import (
    MAX_TOKENS,
    MODEL,
    OUTPUT_CONFIG,
    THINKING,
    require_api_key,
)
from agents.tools import get_financial_profile

MAX_REVISIONS = 2

DRAFT_SYSTEM = """\
You are a personal financial advisor. Work only from the figures given - never
invent balances or accounts. Show the arithmetic behind any number you quote.

Produce three sections: where the user stands, a monthly savings target, and an
investment strategy matched to their risk tolerance.\
"""

REVIEW_SYSTEM = """\
You are a compliance reviewer checking a draft financial plan.

Reply with APPROVED on the first line if the plan: quotes only figures from the
profile, shows its arithmetic, gives a specific monthly savings number, and
matches the stated risk tolerance.

Otherwise reply REVISE on the first line, then list the specific problems.\
"""


class AdvisorState(TypedDict):
    """State threaded through the graph.

    `messages` accumulates via operator.add; every other key is overwritten by
    whichever node returns it.
    """

    messages: Annotated[list[BaseMessage], operator.add]
    user_id: str
    profile: dict
    plan: str
    critique: str
    revisions: int


def _model():
    """Opus 5 rejects temperature - adaptive thinking and effort replace it."""
    require_api_key()
    return ChatAnthropic(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking=THINKING,
        output_config=OUTPUT_CONFIG,
    )


def _text(message):
    """Text out of a response whose content is a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content

    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def load_profile(state: AdvisorState):
    """Deterministic step - pandas, not the model, computes the numbers."""
    profile = get_financial_profile(state.get("user_id", "user123"))
    return {"profile": profile, "revisions": 0}


def draft_plan(state: AdvisorState):
    """Write the plan, or rewrite it against the reviewer's critique."""
    ask = f"Financial profile:\n{state['profile']}\n\nWrite the plan."

    if state.get("critique"):
        ask += (
            f"\n\nYour previous draft was rejected:\n{state['plan']}\n\n"
            f"Reviewer's notes:\n{state['critique']}\n\nRewrite it, fixing these."
        )

    response = _model().invoke([SystemMessage(DRAFT_SYSTEM), HumanMessage(ask)])
    plan = _text(response)

    return {
        "plan": plan,
        "messages": [response],
        "revisions": state.get("revisions", 0) + 1,
    }


def review_plan(state: AdvisorState):
    """Check the draft against the profile it was built from."""
    ask = (
        f"Profile:\n{state['profile']}\n\n"
        f"Draft plan:\n{state['plan']}\n\nReview it."
    )

    response = _model().invoke([SystemMessage(REVIEW_SYSTEM), HumanMessage(ask)])
    return {"critique": _text(response), "messages": [response]}


def should_revise(state: AdvisorState):
    """Conditional edge - the cycle's exit condition."""
    approved = state.get("critique", "").strip().upper().startswith("APPROVED")

    if approved or state.get("revisions", 0) >= MAX_REVISIONS:
        return "done"
    return "revise"


def build_graph():
    """Compile the advisor graph."""
    workflow = StateGraph(AdvisorState)

    workflow.add_node("load_profile", load_profile)
    workflow.add_node("draft_plan", draft_plan)
    workflow.add_node("review_plan", review_plan)

    workflow.set_entry_point("load_profile")
    workflow.add_edge("load_profile", "draft_plan")
    workflow.add_edge("draft_plan", "review_plan")
    workflow.add_conditional_edges(
        "review_plan",
        should_revise,
        {"revise": "draft_plan", "done": END},
    )

    return workflow.compile()


def run_graph_agent(user_id="user123"):
    """Run the graph and return the approved plan."""
    result = build_graph().invoke(
        {"messages": [], "user_id": user_id, "revisions": 0}
    )
    return result["plan"]


if __name__ == "__main__":
    import sys

    from agents.settings import MissingAPIKeyError

    try:
        print(run_graph_agent(sys.argv[1] if len(sys.argv) > 1 else "user123"))
    except MissingAPIKeyError as err:
        sys.exit(f"error: {err}")
