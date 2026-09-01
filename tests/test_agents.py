"""Tests for the agents package: tools, the native loop, and the graph.

The native loop is driven against a fake client that returns scripted
responses, so the control flow - parallel tool use, refusals, truncation,
runaway loops - is tested without spending API calls.
"""

import os
import sys

import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import graph_agent, native_agent
from agents.settings import MissingAPIKeyError
from agents.tools import (
    TOOL_SCHEMAS,
    get_financial_profile,
    project_savings,
    run_tool,
    time_to_goal,
)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


# --- tools --------------------------------------------------------------------

def test_profile_matches_the_mock_data():
    profile = get_financial_profile("user123")

    assert profile["annual_income"] == 70000
    assert profile["monthly_surplus"] == pytest.approx(1800.0)
    assert profile["savings_rate_pct"] == pytest.approx(39.6, abs=0.1)
    assert profile["largest_expense_category"] == "housing"


def test_projection_grows_beyond_contributions():
    result = project_savings(1000, 10, 6.0)

    assert result["total_contributed"] == 120000
    assert result["future_value"] > result["total_contributed"]
    assert result["growth"] == pytest.approx(
        result["future_value"] - result["total_contributed"]
    )


def test_projection_with_zero_return_is_simple_accumulation():
    """The annuity formula divides by the rate - zero must not blow up."""
    assert project_savings(500, 2, 0.0)["future_value"] == 12000


def test_growth_shortens_time_to_goal():
    """31 months with growth vs 34 without, for the same target."""
    with_growth = time_to_goal(60000, 1800, 6.0)["months"]
    without_growth = time_to_goal(60000, 1800, 0.0)["months"]

    assert with_growth < without_growth


def test_tool_errors_come_back_flagged_not_raised():
    output, is_error = run_tool("project_savings", {
        "monthly_contribution": -1, "years": 1, "annual_return_pct": 6,
    })

    assert is_error is True
    assert "ValueError" in output


def test_unknown_tool_is_reported():
    assert run_tool("does_not_exist", {}) == ("Unknown tool: does_not_exist", True)


def test_schemas_are_strict_and_complete():
    """strict:true requires additionalProperties:false and every key required."""
    for schema in TOOL_SCHEMAS:
        assert schema["strict"] is True
        assert schema["input_schema"]["additionalProperties"] is False
        assert set(schema["input_schema"]["required"]) == set(
            schema["input_schema"]["properties"]
        )


# --- native loop --------------------------------------------------------------

def _msg(stop_reason, content):
    return Message(
        id="msg_1", model="claude-opus-5", role="assistant", type="message",
        stop_reason=stop_reason, stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=10), content=content,
    )


class FakeClient:
    """Returns scripted responses and records the messages it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return self._responses.pop(0)


def test_final_answer_skips_thinking_blocks():
    client = FakeClient([
        _msg("end_turn", [
            ThinkingBlock(type="thinking", thinking="reasoning", signature="s"),
            TextBlock(type="text", text="Save $1,800/mo.", citations=None),
        ])
    ])

    assert native_agent.run_native_agent("plan", client=client) == "Save $1,800/mo."


def test_parallel_tool_calls_all_run_in_one_user_message():
    """Two tool_use blocks in one turn must produce two tool_results in a
    single user message - splitting them stops Claude parallelising."""
    client = FakeClient([
        _msg("tool_use", [
            ToolUseBlock(type="tool_use", id="t1", name="get_financial_profile",
                         input={"user_id": "user123"}),
            ToolUseBlock(type="tool_use", id="t2", name="project_savings",
                         input={"monthly_contribution": 1800, "years": 10,
                                "annual_return_pct": 6}),
        ]),
        _msg("end_turn", [TextBlock(type="text", text="Done.", citations=None)]),
    ])

    assert native_agent.run_native_agent("plan", client=client) == "Done."

    # Second request carries the assistant turn plus ONE user message holding
    # both results, in the same order as the calls.
    sent = client.calls[1]
    results = sent[-1]["content"]

    assert sent[-1]["role"] == "user"
    assert len(results) == 2
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"]
    assert all(r["type"] == "tool_result" for r in results)


def test_failing_tool_is_returned_flagged_so_the_model_can_retry():
    client = FakeClient([
        _msg("tool_use", [
            ToolUseBlock(type="tool_use", id="t1", name="project_savings",
                         input={"monthly_contribution": -5, "years": 1,
                                "annual_return_pct": 6}),
        ]),
        _msg("end_turn", [TextBlock(type="text", text="Corrected.", citations=None)]),
    ])

    native_agent.run_native_agent("plan", client=client)
    result = client.calls[1][-1]["content"][0]

    assert result["is_error"] is True
    assert "ValueError" in result["content"]


def test_thinking_blocks_are_echoed_back_on_the_assistant_turn():
    """They must return unchanged for the model to continue its reasoning."""
    thinking = ThinkingBlock(type="thinking", thinking="reasoning", signature="s")
    client = FakeClient([
        _msg("tool_use", [
            thinking,
            ToolUseBlock(type="tool_use", id="t1", name="get_financial_profile",
                         input={"user_id": "user123"}),
        ]),
        _msg("end_turn", [TextBlock(type="text", text="Done.", citations=None)]),
    ])

    native_agent.run_native_agent("plan", client=client)
    assistant_turn = client.calls[1][-2]

    assert assistant_turn["role"] == "assistant"
    assert thinking in assistant_turn["content"]


def test_truncated_response_is_flagged_not_passed_off_as_complete():
    client = FakeClient([
        _msg("max_tokens", [TextBlock(type="text", text="Partial", citations=None)])
    ])

    assert "[truncated" in native_agent.run_native_agent("plan", client=client)


def test_refusal_raises():
    client = FakeClient([_msg("refusal", [])])

    with pytest.raises(RuntimeError, match="declined"):
        native_agent.run_native_agent("plan", client=client)


def test_runaway_tool_loop_terminates():
    """A model that only ever calls tools must not hang the process."""
    tool_turn = _msg("tool_use", [
        ToolUseBlock(type="tool_use", id="t1", name="get_financial_profile",
                     input={"user_id": "user123"}),
    ])
    client = FakeClient([tool_turn] * 5)

    with pytest.raises(RuntimeError, match="did not finish"):
        native_agent.run_native_agent("plan", max_turns=3, client=client)

    assert len(client.calls) == 3


def test_missing_key_raises_before_any_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        native_agent.run_native_agent("plan", client=FakeClient([]))


# --- graph --------------------------------------------------------------------

def test_graph_compiles_with_the_review_cycle():
    graph = graph_agent.build_graph()
    nodes = graph.get_graph().nodes

    for name in ("load_profile", "draft_plan", "review_plan"):
        assert name in nodes


def test_review_approval_ends_the_cycle():
    state = {"critique": "APPROVED - looks right", "revisions": 1}
    assert graph_agent.should_revise(state) == "done"


def test_rejection_sends_it_back_for_a_rewrite():
    state = {"critique": "REVISE - the savings target is missing", "revisions": 1}
    assert graph_agent.should_revise(state) == "revise"


def test_revision_budget_stops_an_endless_rewrite_loop():
    state = {"critique": "REVISE - still wrong", "revisions": graph_agent.MAX_REVISIONS}
    assert graph_agent.should_revise(state) == "done"


def test_load_profile_seeds_state_from_pandas():
    result = graph_agent.load_profile({"user_id": "user123"})

    assert result["revisions"] == 0
    assert result["profile"]["monthly_surplus"] == pytest.approx(1800.0)
