"""Mock financial data for the advisory agent.

Stands in for what would be a real account aggregator (Plaid, a bank export,
a CSV upload). Everything downstream reads from here, so swapping in real data
later means changing only this module.
"""

import pandas as pd

# Keyed by user id so the UI can demo more than one profile.
_USERS = {
    "user123": {
        "name": "Alex Chen",
        "income": 70000,
        "monthly_expenses": {
            "housing": 1500,
            "food": 600,
            "transport": 300,
            "entertainment": 200,
            "others": 150,
        },
        "financial_goals": [
            "save for a house",
            "retire at 60",
        ],
        "risk_tolerance": "moderate",
    },
    "user456": {
        "name": "Jordan Patel",
        "income": 48000,
        "monthly_expenses": {
            "housing": 1250,
            "food": 450,
            "transport": 220,
            "entertainment": 180,
            "others": 200,
        },
        "financial_goals": [
            "pay off student loans",
            "build a 6-month emergency fund",
        ],
        "risk_tolerance": "conservative",
    },
    "user789": {
        "name": "Sam Rivera",
        "income": 145000,
        "monthly_expenses": {
            "housing": 2800,
            "food": 900,
            "transport": 500,
            "entertainment": 650,
            "others": 400,
        },
        "financial_goals": [
            "max out retirement accounts",
            "start a side business",
            "buy a rental property",
        ],
        "risk_tolerance": "aggressive",
    },
}

# Rough effective tax rate. Real code would use brackets; this keeps the mock
# honest enough that the take-home numbers are not wildly optimistic.
EFFECTIVE_TAX_RATE = 0.22


def list_users():
    """Return {user_id: display_name} for every mock profile."""
    return {user_id: data["name"] for user_id, data in _USERS.items()}


def get_user_financial_data(user_id="user123"):
    """Return one user's financial profile.

    Falls back to the default profile for an unknown id so the UI never
    crashes on a stale selection.
    """
    return _USERS.get(user_id, _USERS["user123"]).copy()


def get_expense_dataframe(user_data):
    """Build a pandas DataFrame of monthly expenses.

    Columns: category, amount, pct_of_spending, pct_of_take_home.
    """
    monthly_take_home = get_monthly_take_home(user_data)
    expenses = user_data["monthly_expenses"]

    df = pd.DataFrame(
        sorted(expenses.items(), key=lambda item: item[1], reverse=True),
        columns=["category", "amount"],
    )
    total = df["amount"].sum()

    df["pct_of_spending"] = (df["amount"] / total * 100).round(1)
    df["pct_of_take_home"] = (df["amount"] / monthly_take_home * 100).round(1)

    return df


def get_monthly_take_home(user_data):
    """Estimated monthly income after tax."""
    return user_data["income"] * (1 - EFFECTIVE_TAX_RATE) / 12


def summarize_finances(user_data):
    """Compute the headline numbers the agent and the UI both need."""
    df = get_expense_dataframe(user_data)
    monthly_take_home = get_monthly_take_home(user_data)
    total_expenses = float(df["amount"].sum())
    surplus = monthly_take_home - total_expenses

    return {
        "annual_income": user_data["income"],
        "monthly_gross": user_data["income"] / 12,
        "monthly_take_home": monthly_take_home,
        "total_monthly_expenses": total_expenses,
        "monthly_surplus": surplus,
        "savings_rate": surplus / monthly_take_home * 100,
        "largest_category": df.iloc[0]["category"],
        "expenses": df,
    }


def format_expenses_for_prompt(user_data):
    """Render the expense breakdown as a compact table for the LLM prompt.

    Giving the model the derived percentages and totals - rather than raw
    category amounts - keeps it from doing (and fumbling) the arithmetic.
    """
    summary = summarize_finances(user_data)
    df = summary["expenses"]

    lines = [
        f"- {row.category}: ${row.amount:,.0f}/mo "
        f"({row.pct_of_take_home:.1f}% of take-home pay)"
        for row in df.itertuples()
    ]
    lines.append(
        f"- TOTAL: ${summary['total_monthly_expenses']:,.0f}/mo"
    )
    lines.append(
        f"\nEstimated monthly take-home (after ~{EFFECTIVE_TAX_RATE:.0%} tax): "
        f"${summary['monthly_take_home']:,.0f}"
    )
    lines.append(
        f"Monthly surplus: ${summary['monthly_surplus']:,.0f} "
        f"(savings rate {summary['savings_rate']:.1f}%)"
    )

    return "\n".join(lines)
