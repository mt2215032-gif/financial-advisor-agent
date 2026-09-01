"""
CrewAI Financial Advisor Agent
"""

from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from agents.settings import MAX_TOKENS, MODEL, require_api_key
from agents.tools import (
    get_financial_profile,
    project_savings,
    time_to_goal,
)

CREW_MODEL = f"anthropic/{MODEL}"









"""
CrewAI team: analyst -> planner -> writer, running on Claude.

This module requires CrewAI to be installed in requirements.txt.

Example:
    crewai>=1.15,<2.0
"""

try:
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool

except Exception as err:
    raise ImportError(
        "CrewAI import failed.\n\n"
        f"Original error: {type(err).__name__}: {err}\n\n"
        "Make sure your requirements.txt contains:\n"
        "crewai>=1.15,<2.0"
    ) from err


from agents.settings import MAX_TOKENS, MODEL, require_api_key
from agents.tools import (
    get_financial_profile,
    project_savings,
    time_to_goal,
)


# CrewAI requires a provider prefix for the model.
CREW_MODEL = f"anthropic/{MODEL}"


# ============================================================
# TOOLS
# ============================================================

@tool("Get financial profile")
def profile_tool(user_id: str) -> str:
    """
    Look up the user's income, expenses, goals,
    risk tolerance, and monthly financial figures.
    """

    return str(get_financial_profile(user_id))


@tool("Project savings growth")
def projection_tool(
    monthly_contribution: float,
    years: float,
    annual_return_pct: float = 6.0,
) -> str:
    """
    Project the future value of monthly contributions
    using monthly compounding.
    """

    return str(
        project_savings(
            monthly_contribution,
            years,
            annual_return_pct,
        )
    )


@tool("Time to reach a goal")
def goal_tool(
    target_amount: float,
    monthly_contribution: float,
    annual_return_pct: float = 6.0,
) -> str:
    """
    Calculate how many months are needed to reach
    a financial target.
    """

    return str(
        time_to_goal(
            target_amount,
            monthly_contribution,
            annual_return_pct,
        )
    )


# ============================================================
# LLM
# ============================================================

def build_llm():
    """
    Configure Claude for CrewAI.
    """

    require_api_key()

    return LLM(
        model=CREW_MODEL,
        max_tokens=MAX_TOKENS,
    )


# ============================================================
# CREW
# ============================================================

def build_crew(user_id="user123"):
    """
    Build the three-agent financial advisory crew.
    """

    claude = build_llm()


    # --------------------------------------------------------
    # AGENT 1: FINANCIAL ANALYST
    # --------------------------------------------------------

    analyst = Agent(
        role="Financial Analyst",

        goal=(
            "Analyze the user's financial situation accurately "
            "using the financial profile tool."
        ),

        backstory=(
            "You are a careful financial analyst who verifies "
            "all numbers before reporting them."
        ),

        tools=[
            profile_tool
        ],

        llm=claude,

        verbose=True,
    )


    # --------------------------------------------------------
    # AGENT 2: FINANCIAL PLANNER
    # --------------------------------------------------------

    planner = Agent(
        role="Retirement and Savings Planner",

        goal=(
            "Create a realistic savings plan and general "
            "investment strategy based on the user's goals "
            "and risk tolerance."
        ),

        backstory=(
            "You are an experienced financial planner who focuses "
            "on emergency savings, financial goals, and realistic "
            "long-term planning."
        ),

        tools=[
            projection_tool,
            goal_tool,
        ],

        llm=claude,

        verbose=True,
    )


    # --------------------------------------------------------
    # AGENT 3: WRITER
    # --------------------------------------------------------

    writer = Agent(
        role="Client Communications Lead",

        goal=(
            "Convert the financial analysis into a simple, clear, "
            "and actionable financial plan."
        ),

        backstory=(
            "You explain financial concepts in simple language "
            "without unnecessary jargon."
        ),

        llm=claude,

        verbose=True,
    )


    # ========================================================
    # TASK 1: ANALYSIS
    # ========================================================

    analyse = Task(
        description=(
            f"Use the financial profile tool to retrieve the "
            f"financial information for user '{user_id}'.\n\n"

            "Calculate and report:\n"
            "- Monthly income\n"
            "- Total monthly expenses\n"
            "- Monthly surplus\n"
            "- Savings rate\n"
            "- Largest expense category\n\n"

            "Use only numbers returned by the financial profile tool."
        ),

        expected_output=(
            "A markdown summary of the user's current financial "
            "situation with clear calculations."
        ),

        agent=analyst,
    )


    # ========================================================
    # TASK 2: PLANNING
    # ========================================================

    plan = Task(
        description=(
            "Using the financial analyst's results, create a "
            "financial savings plan.\n\n"

            "You must:\n"
            "1. Recommend a monthly savings target.\n"
            "2. Include an emergency fund strategy.\n"
            "3. Allocate savings across the user's financial goals.\n"
            "4. Use the projection tool for future-value calculations.\n"
            "5. Use the goal tool when calculating timelines.\n"
            "6. Suggest a general investment allocation based on "
            "the user's risk tolerance.\n\n"

            "Do not guarantee investment returns."
        ),

        expected_output=(
            "A financial plan including monthly savings targets, "
            "goal allocations, timelines, and a general investment "
            "allocation."
        ),

        agent=planner,

        context=[
            analyse
        ],
    )


    # ========================================================
    # TASK 3: FINAL RESPONSE
    # ========================================================

    write = Task(
        description=(
            "Write the final financial plan for the user.\n\n"

            "Use exactly these three sections:\n\n"

            "## 1. Where You Stand\n"
            "Summarize the user's current financial situation.\n\n"

            "## 2. Monthly Savings Goal\n"
            "Explain how much the user should consider saving "
            "and how it could be divided between goals.\n\n"

            "## 3. General Investment Strategy\n"
            "Explain an educational investment strategy based "
            "on the user's risk tolerance.\n\n"

            "End with this disclaimer:\n"
            "'This information is for educational purposes and "
            "is not personalized advice from a licensed financial "
            "professional.'"
        ),

        expected_output=(
            "A clean and easy-to-read three-section markdown "
            "financial plan."
        ),

        agent=writer,

        context=[
            analyse,
            plan,
        ],
    )


    # ========================================================
    # RETURN CREW
    # ========================================================

    return Crew(
        agents=[
            analyst,
            planner,
            writer,
        ],

        tasks=[
            analyse,
            plan,
            write,
        ],

        process=Process.sequential,

        verbose=True,
    )


# ============================================================
# RUN CREW
# ============================================================

def run_crew_agent(user_id="user123"):
    """
    Run the complete financial advisory crew.
    """

    crew = build_crew(user_id)

    result = crew.kickoff()

    return str(result)


# ============================================================
# LOCAL TESTING
# ============================================================

if __name__ == "__main__":

    import sys

    from agents.settings import MissingAPIKeyError

    try:

        user_id = (
            sys.argv[1]
            if len(sys.argv) > 1
            else "user123"
        )

        print(
            run_crew_agent(user_id)
        )

    except MissingAPIKeyError as err:

        sys.exit(
            f"Error: {err}"
        )
