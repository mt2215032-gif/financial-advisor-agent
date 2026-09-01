"""Financial tools the agents can call.

Deterministic money math lives here, not in the model. Each function has a
matching JSON schema in TOOL_SCHEMAS so the same implementations back the
native SDK loop and any framework that takes tool definitions.
"""

import math

from mock_data import get_user_financial_data, list_users, summarize_finances


def get_financial_profile(user_id="user123"):
    """Look up a user's profile and its derived monthly figures."""
    user_data = get_user_financial_data(user_id)
    summary = summarize_finances(user_data)

    return {
        "name": user_data["name"],
        "annual_income": user_data["income"],
        "risk_tolerance": user_data["risk_tolerance"],
        "financial_goals": user_data["financial_goals"],
        "monthly_take_home": round(summary["monthly_take_home"], 2),
        "monthly_expenses": user_data["monthly_expenses"],
        "total_monthly_expenses": round(summary["total_monthly_expenses"], 2),
        "monthly_surplus": round(summary["monthly_surplus"], 2),
        "savings_rate_pct": round(summary["savings_rate"], 1),
        "largest_expense_category": summary["largest_category"],
    }


def project_savings(monthly_contribution, years, annual_return_pct=6.0):
    """Future value of a monthly contribution, compounded monthly.

    Standard future-value-of-an-annuity formula. A 0% return degrades to
    simple accumulation rather than dividing by zero.
    """
    if monthly_contribution < 0 or years <= 0:
        raise ValueError("monthly_contribution must be >= 0 and years > 0")

    months = int(round(years * 12))
    monthly_rate = annual_return_pct / 100 / 12

    if monthly_rate == 0:
        future_value = monthly_contribution * months
    else:
        future_value = monthly_contribution * (
            ((1 + monthly_rate) ** months - 1) / monthly_rate
        )

    contributed = monthly_contribution * months

    return {
        "future_value": round(future_value, 2),
        "total_contributed": round(contributed, 2),
        "growth": round(future_value - contributed, 2),
        "months": months,
        "annual_return_pct": annual_return_pct,
    }


def time_to_goal(target_amount, monthly_contribution, annual_return_pct=6.0):
    """How long a monthly contribution takes to reach a target."""
    if monthly_contribution <= 0:
        raise ValueError("monthly_contribution must be > 0")
    if target_amount <= 0:
        raise ValueError("target_amount must be > 0")

    monthly_rate = annual_return_pct / 100 / 12

    if monthly_rate == 0:
        months = target_amount / monthly_contribution
    else:
        # Annuity formula solved for the number of periods.
        months = math.log(
            1 + (target_amount * monthly_rate) / monthly_contribution
        ) / math.log(1 + monthly_rate)

    months = math.ceil(months)

    return {
        "months": months,
        "years": round(months / 12, 1),
        "target_amount": target_amount,
        "monthly_contribution": monthly_contribution,
        "annual_return_pct": annual_return_pct,
    }


# Name -> implementation, used to dispatch a tool_use block.
TOOL_FUNCTIONS = {
    "get_financial_profile": get_financial_profile,
    "project_savings": project_savings,
    "time_to_goal": time_to_goal,
}

# `strict: true` guarantees the arguments validate exactly against the schema,
# which requires additionalProperties: false and an explicit required list.
TOOL_SCHEMAS = [
    {
        "name": "get_financial_profile",
        "description": (
            "Look up a user's income, monthly expenses, goals, risk tolerance, "
            "and derived figures (take-home pay, surplus, savings rate). Call "
            "this before giving any advice so the numbers are real."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": f"One of: {', '.join(list_users())}",
                }
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "project_savings",
        "description": (
            "Project what a monthly contribution grows to over a number of "
            "years, compounded monthly. Use for retirement and savings targets."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "monthly_contribution": {
                    "type": "number",
                    "description": "Dollars contributed each month.",
                },
                "years": {
                    "type": "number",
                    "description": "How many years to project.",
                },
                "annual_return_pct": {
                    "type": "number",
                    "description": "Expected annual return, percent. Default 6.",
                },
            },
            "required": ["monthly_contribution", "years", "annual_return_pct"],
            "additionalProperties": False,
        },
    },
    {
        "name": "time_to_goal",
        "description": (
            "Compute how many months of a given monthly contribution are "
            "needed to reach a target amount, including growth."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_amount": {
                    "type": "number",
                    "description": "The dollar amount to reach.",
                },
                "monthly_contribution": {
                    "type": "number",
                    "description": "Dollars contributed each month.",
                },
                "annual_return_pct": {
                    "type": "number",
                    "description": "Expected annual return, percent. Default 6.",
                },
            },
            "required": [
                "target_amount", "monthly_contribution", "annual_return_pct",
            ],
            "additionalProperties": False,
        },
    },
]


def run_tool(name, tool_input):
    """Dispatch one tool call. Returns (result_text, is_error)."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}", True

    try:
        return str(func(**tool_input)), False
    except Exception as err:
        # Handing the error back lets the model correct its arguments and
        # retry, which is far more useful than crashing the loop.
        return f"{type(err).__name__}: {err}", True
