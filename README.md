# Individualized Financial Advisory Agent 💰

A chat-based financial advisor built with **LangChain**, **Claude**, **Streamlit**,
and **pandas**. It reads a user's financial profile, computes the real numbers with
pandas, and asks Claude for a personalized plan — a status summary, a monthly
savings target, and an investment strategy matched to the user's risk tolerance —
then answers follow-up questions about that plan.

> Educational demo on mock data. Not licensed financial advice.

---

## Quick start

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env              # then add your ANTHROPIC_API_KEY

streamlit run app.py
```

Get an API key at [console.anthropic.com](https://console.anthropic.com/settings/keys).

To run the agent without the UI:

```bash
python advisor_agent.py
```

---

## How it works

```
mock_data.py ──► pandas summary ──► advisor_prompt.py ──► advisor_agent.py ──► app.py
  profiles        the arithmetic       prompt templates      LangChain + Claude    Streamlit UI
```

| File | Responsibility |
| --- | --- |
| `mock_data.py` | Three mock profiles plus the pandas layer that derives take-home pay, the expense breakdown, surplus, and savings rate. Swap this module for a real data source (bank export, Plaid) and nothing downstream changes. |
| `advisor_prompt.py` | Two `ChatPromptTemplate`s — one for the initial plan, one for follow-up questions — sharing a single advisor persona. |
| `advisor_agent.py` | Builds the `ChatAnthropic` model, renders the prompts, and exposes blocking and streaming calls. |
| `app.py` | Streamlit UI: profile picker, metrics, expense chart, streamed plan, follow-up chat. |
| `tests/test_advisor.py` | Tests for the arithmetic and prompt assembly. No API key needed. |

**The arithmetic happens in pandas, not in the model.** `format_expenses_for_prompt()`
hands Claude a pre-computed table — each category as a share of take-home pay, the
total, the surplus, the savings rate — so the model spends its effort on
recommendations instead of on mental math it might get wrong.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key. |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Model to use. |
| `ANTHROPIC_EFFORT` | `medium` | Reasoning depth: `low`, `medium`, `high`, `xhigh`, `max`. |

Claude Opus 5 rejects `temperature`, `top_p`, and `top_k`, so response depth is
controlled with adaptive thinking plus the effort level rather than a sampling
temperature. Because adaptive thinking returns a list of content blocks rather
than a bare string, `_extract_text()` pulls out the text blocks.

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -q
```

---

## Extending it

- **Real data** — replace `get_user_financial_data()` with a bank or aggregator
  client; the pandas helpers and prompts work unchanged.
- **More profiles** — add an entry to `_USERS` in `mock_data.py`; it appears in the
  sidebar automatically.
- **Debt and net worth** — extend the profile dict, surface it in
  `summarize_finances()`, and add a line to `format_expenses_for_prompt()`.
- **Tool use** — give the agent live rate or market-data tools so allocation advice
  reflects current conditions.
