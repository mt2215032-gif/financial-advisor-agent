"""Tests for the data layer and prompt assembly.

These run without an API key - nothing here calls Claude.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor_agent import _extract_text, _format_history, _profile_variables
from advisor_prompt import advisor_template, followup_template
from mock_data import (
    format_expenses_for_prompt,
    get_expense_dataframe,
    get_monthly_take_home,
    get_user_financial_data,
    list_users,
    summarize_finances,
)


@pytest.fixture
def user():
    return get_user_financial_data("user123")


# --- mock_data ----------------------------------------------------------------

def test_default_profile_matches_lab_spec(user):
    assert user["income"] == 70000
    assert user["monthly_expenses"]["housing"] == 1500
    assert user["risk_tolerance"] == "moderate"
    assert "save for a house" in user["financial_goals"]


def test_unknown_user_falls_back_to_default():
    assert get_user_financial_data("nope") == get_user_financial_data("user123")


def test_every_listed_user_resolves():
    for user_id in list_users():
        assert get_user_financial_data(user_id)["income"] > 0


def test_expense_dataframe_is_sorted_and_adds_up(user):
    df = get_expense_dataframe(user)

    assert list(df.columns) == [
        "category", "amount", "pct_of_spending", "pct_of_take_home",
    ]
    assert df["amount"].is_monotonic_decreasing
    assert df["amount"].sum() == sum(user["monthly_expenses"].values())
    assert df["pct_of_spending"].sum() == pytest.approx(100, abs=0.5)


def test_summary_arithmetic(user):
    summary = summarize_finances(user)

    assert summary["monthly_take_home"] == pytest.approx(70000 * 0.78 / 12)
    assert summary["total_monthly_expenses"] == 2750
    assert summary["monthly_surplus"] == pytest.approx(
        summary["monthly_take_home"] - 2750
    )
    assert summary["savings_rate"] == pytest.approx(
        summary["monthly_surplus"] / summary["monthly_take_home"] * 100
    )
    assert summary["largest_category"] == "housing"


def test_take_home_is_below_gross(user):
    assert get_monthly_take_home(user) < user["income"] / 12


def test_prompt_expense_block_carries_the_math(user):
    block = format_expenses_for_prompt(user)

    assert "housing" in block
    assert "TOTAL" in block
    assert "$2,750" in block
    assert "Monthly surplus" in block


# --- prompts ------------------------------------------------------------------

def test_advisor_template_variables():
    assert set(advisor_template.input_variables) == {
        "income", "expenses", "goals", "risk",
    }


def test_followup_template_variables():
    assert set(followup_template.input_variables) == {
        "income", "expenses", "goals", "risk", "history", "question",
    }


def test_advisor_prompt_renders_the_profile(user):
    messages = advisor_template.format_messages(**_profile_variables(user))
    system, human = messages

    assert "financial advisor" in system.content
    assert "$70,000" in human.content
    assert "moderate" in human.content
    assert "save for a house" in human.content
    assert "Monthly Savings Goal" in human.content


def test_followup_prompt_includes_history(user):
    messages = followup_template.format_messages(
        question="Can I afford a car?",
        history=_format_history([("user", "Hi"), ("assistant", "Hello")]),
        **_profile_variables(user),
    )

    assert "Can I afford a car?" in messages[-1].content
    assert "User: Hi" in messages[-1].content
    assert "Advisor: Hello" in messages[-1].content


# --- agent helpers ------------------------------------------------------------

def test_extract_text_from_plain_string():
    assert _extract_text("hello") == "hello"


def test_extract_text_skips_thinking_blocks():
    content = [
        {"type": "thinking", "thinking": "internal reasoning"},
        {"type": "text", "text": "Here is "},
        {"type": "text", "text": "your plan."},
    ]
    assert _extract_text(content) == "Here is your plan."


def test_format_history_empty():
    assert _format_history([]) == "(no prior questions)"


def test_format_history_keeps_only_recent_turns():
    history = [("user", f"q{i}") for i in range(10)]
    rendered = _format_history(history, limit=3)

    assert "q9" in rendered
    assert "q0" not in rendered
