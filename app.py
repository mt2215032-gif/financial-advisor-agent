"""Streamlit UI for the Individualized Financial Advisory Agent."""

import os

import streamlit as st

from advisor_agent import (
    MissingAPIKeyError,
    stream_financial_advisor,
    stream_followup,
)
from agents.providers import (
    PROVIDERS,
    UnknownProviderError,
    get_provider,
    model_name_for,
)
from mock_data import get_user_financial_data, list_users, summarize_finances

st.set_page_config(page_title="Financial Advisory Agent", page_icon="💰")

# Streamlit's dataframe row/header heights, used to size the table to its rows
# exactly so it lines up with the chart beside it and shows no blank filler row.
ROW_HEIGHT = 35
HEADER_HEIGHT = 38

st.title("Individualized Financial Advisory Agent 💰")
st.caption("Personalized savings, budget, and investment guidance from your profile.")


# --- Sidebar: pick whose finances we are looking at ---------------------------

users = list_users()

with st.sidebar:
    st.header("Profile")
    user_id = st.selectbox(
        "User",
        options=list(users),
        format_func=lambda uid: f"{users[uid]} ({uid})",
    )
    st.divider()
    st.header("Model")

    provider_keys = list(PROVIDERS)
    try:
        active = get_provider().key
    except UnknownProviderError as err:
        # A typo in LLM_PROVIDER should not take the whole app down.
        st.error(str(err))
        st.stop()

    provider_key = st.selectbox(
        "Provider",
        options=provider_keys,
        index=provider_keys.index(active),
        format_func=lambda key: PROVIDERS[key].label,
        help="Set the default with LLM_PROVIDER in .env",
    )
    provider = PROVIDERS[provider_key]
    st.caption(f"Model: `{model_name_for(provider)}`")

    if provider.env_key and not os.getenv(provider.env_key):
        st.warning(f"{provider.env_key} is not set.", icon="⚠️")
    elif provider.env_key is None:
        st.caption("Runs locally — no API key needed.")

# Switching profiles invalidates the plan and the conversation about it.
if (st.session_state.get("user_id"), st.session_state.get("provider")) != (
    user_id, provider_key
):
    st.session_state.user_id = user_id
    st.session_state.provider = provider_key
    st.session_state.plan = None
    st.session_state.history = []

st.session_state.setdefault("plan", None)
st.session_state.setdefault("history", [])

user_data = get_user_financial_data(user_id)
summary = summarize_finances(user_data)


# --- Financial snapshot -------------------------------------------------------

st.subheader(f"{user_data['name']}'s Snapshot")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Take-home / mo", f"${summary['monthly_take_home']:,.0f}")
col2.metric("Expenses / mo", f"${summary['total_monthly_expenses']:,.0f}")
col3.metric(
    "Surplus / mo",
    f"${summary['monthly_surplus']:,.0f}",
    delta=f"{summary['savings_rate']:.1f}% savings rate",
)
col4.metric("Risk tolerance", user_data["risk_tolerance"].title())

with st.expander("Expense breakdown", expanded=True):
    panel_height = HEADER_HEIGHT + ROW_HEIGHT * len(summary["expenses"])
    breakdown, chart = st.columns([3, 2])
    with breakdown:
        st.dataframe(
            summary["expenses"],
            hide_index=True,
            column_config={
                "category": "Category",
                "amount": st.column_config.NumberColumn("Amount", format="dollar"),
                "pct_of_spending": st.column_config.NumberColumn(
                    "% of spending", format="%.1f%%"
                ),
                "pct_of_take_home": st.column_config.NumberColumn(
                    "% of take-home", format="%.1f%%"
                ),
            },
            width="stretch",
            height=panel_height,
        )
    with chart:
        st.bar_chart(
            summary["expenses"].set_index("category")["amount"],
            height=panel_height,
        )

st.write("**Goals:** " + ", ".join(user_data["financial_goals"]))


# --- The plan -----------------------------------------------------------------

st.divider()

if st.button("Get Advice", type="primary"):
    st.session_state.history = []
    try:
        with st.chat_message("assistant"):
            st.session_state.plan = st.write_stream(
                stream_financial_advisor(user_data=user_data, provider=provider_key)
            )
    except MissingAPIKeyError as err:
        st.error(str(err))
    except Exception as err:  # network failure, rate limit, bad model id
        st.error(f"Could not reach {provider.label}: {err}")

elif st.session_state.plan:
    with st.chat_message("assistant"):
        st.markdown(st.session_state.plan)


# --- Follow-up chat -----------------------------------------------------------

if st.session_state.plan:
    st.divider()
    st.subheader("Ask a follow-up")

    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    question = st.chat_input("e.g. What if I want to buy a house in 3 years?")
    if question:
        st.session_state.history.append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.chat_message("assistant"):
                answer = st.write_stream(
                    stream_followup(
                        question,
                        # Drop the turn we just appended - it is passed
                        # separately as the question.
                        history=st.session_state.history[:-1],
                        user_data=user_data,
                        provider=provider_key,
                    )
                )
            st.session_state.history.append(("assistant", answer))
        except MissingAPIKeyError as err:
            st.error(str(err))
        except Exception as err:
            st.error(f"Could not reach {provider.label}: {err}")

st.divider()
st.caption(
    "Educational demo built on mock data — not licensed financial advice."
)
