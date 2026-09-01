from langchain_anthropic import ChatAnthropic

from advisor_prompt import advisor_template
from mock_data import get_user_financial_data


def run_financial_advisor():

    # Get user financial data
    user_data = get_user_financial_data()

    # Format expenses
    expenses = "\n".join(
        f"{category}: ${amount}"
        for category, amount
        in user_data["monthly_expenses"].items()
    )

    # Create messages from prompt template
    messages = advisor_template.format_messages(
        income=user_data["income"],
        expenses=expenses,
        goals=", ".join(user_data["financial_goals"]),
        risk=user_data["risk_tolerance"]
    )

    # Initialize Claude
    llm = ChatAnthropic(
        model="claude-3-5-haiku-latest",
        temperature=0.4,
        max_tokens=1000
    )

    # Get AI response
    response = llm.invoke(messages)

    return response.content