"""CrewAI team: analyst -> planner -> writer, running on Claude.

Two corrections to the usual version of this template:

1. Current CrewAI does not take a LangChain `ChatAnthropic` object for `llm`.
   It has its own `LLM` class, and the model id needs a provider prefix -
   "anthropic/claude-opus-5".
2. `temperature=0.2` is forwarded to the API whenever it is set, and Opus 5
   rejects temperature with a 400. Leaving it unset is what makes this work;
   depth comes from the model's adaptive thinking instead.

Requires the optional extras:  pip install -r requirements-agents.txt
"""

try:
    from crewai import LLM, Agent, Crew, Process, Task
    from crewai.tools import tool
except Exception as err:
    raise ImportError(
        f"CrewAI import failed.\n"
        f"Original error: {type(err).__name__}: {err}\n\n"
        f"Make sure CrewAI is installed in requirements.txt."
    ) from err


# --- tools ---------------------------------------------------------------------
# Thin wrappers so the crew shares the same deterministic math as every other
# implementation here.

@tool("Get financial profile")
def profile_tool(user_id: str) -> str:
    """Look up a user's income, expenses, goals, risk tolerance, and derived
    monthly figures. Call this before giving advice."""
    return str(get_financial_profile(user_id))


@tool("Project savings growth")
def projection_tool(
    monthly_contribution: float, years: float, annual_return_pct: float = 6.0
) -> str:
    """Project what a monthly contribution grows to over a number of years,
    compounded monthly."""
    return str(project_savings(monthly_contribution, years, annual_return_pct))


@tool("Time to reach a goal")
def goal_tool(
    target_amount: float, monthly_contribution: float, annual_return_pct: float = 6.0
) -> str:
    """Compute how many months of a monthly contribution are needed to reach a
    target amount, including growth."""
    return str(time_to_goal(target_amount, monthly_contribution, annual_return_pct))


def build_llm():
    """Claude, configured for CrewAI.

    No temperature: Opus 5 rejects sampling parameters.
    """
    require_api_key()
    return LLM(model=CREW_MODEL, max_tokens=MAX_TOKENS)


def build_crew(user_id="user123"):
    """Assemble the three-agent advisory crew."""
    claude = build_llm()

    analyst = Agent(
        role="Financial Analyst",
        goal=(
            "Establish exactly what a user's finances look like, using the "
            "tools rather than estimating."
        ),
        backstory=(
            "A former auditor who refuses to quote a number they have not "
            "verified, and who always shows the arithmetic behind a figure."
        ),
        tools=[profile_tool],
        llm=claude,
        verbose=True,
    )

    planner = Agent(
        role="Retirement and Savings Planner",
        goal=(
            "Turn the analyst's figures into a monthly savings target and an "
            "allocation matched to the user's risk tolerance."
        ),
        backstory=(
            "A CFP who sizes an emergency fund first, then splits what is left "
            "across the user's stated goals with explicit timelines."
        ),
        tools=[projection_tool, goal_tool],
        llm=claude,
        verbose=True,
    )

    writer = Agent(
        role="Client Communications Lead",
        goal="Deliver the plan in plain language the client can act on.",
        backstory=(
            "Writes for people who are not financial professionals: short "
            "sentences, concrete dollar amounts, no jargon."
        ),
        llm=claude,
        verbose=True,
    )

    analyse = Task(
        description=(
            f"Use the profile tool to pull the finances for user '{user_id}'. "
            "Report take-home pay, total expenses, surplus, savings rate, and "
            "the largest expense category. Quote only figures the tool returned."
        ),
        expected_output=(
            "A markdown summary of the user's monthly position, with the "
            "arithmetic shown for the surplus and savings rate."
        ),
        agent=analyst,
    )

    plan = Task(
        description=(
            "From the analyst's figures, set a monthly savings target and split "
            "it across an emergency fund and the user's goals. Use the "
            "projection and goal tools for every forward-looking number. "
            "Recommend an allocation matched to the stated risk tolerance."
        ),
        expected_output=(
            "A savings target in dollars, its split across goals with "
            "timelines, and an allocation with percentages."
        ),
        agent=planner,
        context=[analyse],
    )

    write = Task(
        description=(
            "Write the final client-facing plan in three sections: where they "
            "stand, their monthly savings goal, and their investment strategy. "
            "End with one line noting this is educational, not licensed advice."
        ),
        expected_output="A clean three-section markdown plan.",
        agent=writer,
        context=[analyse, plan],
    )

    return Crew(
        agents=[analyst, planner, writer],
        tasks=[analyse, plan, write],
        process=Process.sequential,
        verbose=True,
    )


def run_crew_agent(user_id="user123"):
    """Run the crew and return the final written plan."""
    return str(build_crew(user_id).kickoff())


if __name__ == "__main__":
    import sys

    from agents.settings import MissingAPIKeyError

    try:
        print(run_crew_agent(sys.argv[1] if len(sys.argv) > 1 else "user123"))
    except MissingAPIKeyError as err:
        sys.exit(f"error: {err}")
