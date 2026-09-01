"""Ask the advisor questions from the terminal and get real answers.

    python -m agents.chat                 # default profile
    python -m agents.chat user456         # a different profile

Prints the plan, then answers follow-up questions until you type 'exit'.
Answers stream in as they are generated. Works with whichever provider
LLM_PROVIDER selects.
"""

import sys

from advisor_agent import stream_financial_advisor, stream_followup
from agents.providers import (
    MissingCredentialError,
    UnknownProviderError,
    get_provider,
    model_name_for,
)
from mock_data import get_user_financial_data, list_users, summarize_finances

BANNER = "=" * 68


def _stream(chunks):
    """Print chunks as they arrive, and return the joined text."""
    parts = []
    for chunk in chunks:
        parts.append(chunk)
        print(chunk, end="", flush=True)
    print()
    return "".join(parts)


def main(argv):
    user_id = argv[0] if argv else "user123"

    if user_id not in list_users():
        sys.exit(
            f"error: unknown user '{user_id}'. "
            f"Choose one of: {', '.join(list_users())}"
        )

    try:
        provider = get_provider()
    except UnknownProviderError as err:
        sys.exit(f"error: {err}")

    user_data = get_user_financial_data(user_id)
    summary = summarize_finances(user_data)

    print(BANNER)
    print(f"Financial advisor - {user_data['name']} ({user_id})")
    print(f"{provider.label} · {model_name_for(provider)}")
    print(BANNER)
    print(
        f"Take-home ${summary['monthly_take_home']:,.0f}/mo · "
        f"expenses ${summary['total_monthly_expenses']:,.0f} · "
        f"surplus ${summary['monthly_surplus']:,.0f} "
        f"({summary['savings_rate']:.1f}% savings rate)"
    )
    print(f"Goals: {', '.join(user_data['financial_goals'])}")
    print(f"Risk tolerance: {user_data['risk_tolerance']}\n")

    print("Generating plan...\n")
    try:
        _stream(stream_financial_advisor(user_data=user_data))
    except MissingCredentialError as err:
        sys.exit(f"\nerror: {err}")
    except Exception as err:
        sys.exit(f"\nerror: could not reach {provider.label}: {err}")

    history = []
    print("\n" + BANNER)
    print("Ask a follow-up, or type 'exit' to quit.")
    print(BANNER)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        print()
        try:
            answer = _stream(
                stream_followup(question, history=history, user_data=user_data)
            )
        except Exception as err:
            print(f"error: {err}")
            continue

        history.append(("user", question))
        history.append(("assistant", answer))

    print("Bye.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
