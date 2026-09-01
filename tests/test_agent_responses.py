"""Tests for response handling, against mocked Anthropic API responses.

Adaptive thinking makes the model return a list of content blocks rather than
a plain string, so these exercise the parsing path end to end - without
spending an API call.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    Usage,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import advisor_agent
from advisor_agent import (
    MissingAPIKeyError,
    answer_followup,
    build_llm,
    run_financial_advisor,
    stream_financial_advisor,
    stream_followup,
)

PLAN_TEXT = "### 1. Where You Stand\nYou save $1,800/mo."


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


def _message():
    """A response carrying a thinking block ahead of the answer."""
    return Message(
        id="msg_1",
        model="claude-opus-5",
        role="assistant",
        type="message",
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=20),
        content=[
            ThinkingBlock(
                type="thinking", thinking="internal reasoning", signature="sig"
            ),
            TextBlock(type="text", text=PLAN_TEXT, citations=None),
        ],
    )


def _stream_events():
    """The same response as a stream of server-sent events."""
    yield RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_1",
            model="claude-opus-5",
            role="assistant",
            type="message",
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=0),
            content=[],
        ),
    )

    yield RawContentBlockStartEvent(
        type="content_block_start",
        index=0,
        content_block=ThinkingBlock(type="thinking", thinking="", signature=""),
    )
    yield RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=0,
        delta=ThinkingDelta(type="thinking_delta", thinking="reasoning..."),
    )
    yield RawContentBlockStopEvent(type="content_block_stop", index=0)

    yield RawContentBlockStartEvent(
        type="content_block_start",
        index=1,
        content_block=TextBlock(type="text", text="", citations=None),
    )
    for piece in ["### 1. Where ", "You Stand\n", "You save $1,800/mo."]:
        yield RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=1,
            delta=TextDelta(type="text_delta", text=piece),
        )
    yield RawContentBlockStopEvent(type="content_block_stop", index=1)

    yield RawMessageDeltaEvent(
        type="message_delta",
        delta={"stop_reason": "end_turn", "stop_sequence": None},
        usage={"output_tokens": 20},
    )
    yield RawMessageStopEvent(type="message_stop")


def _patch_api(payload):
    """Patch the SDK call ChatAnthropic makes, returning `payload`."""
    raw = MagicMock()
    raw.parse.return_value = payload
    return patch.object(type(build_llm()), "_create", return_value=raw)


# --- non-streaming ------------------------------------------------------------

def test_plan_drops_thinking_and_returns_text():
    with _patch_api(_message()):
        assert run_financial_advisor() == PLAN_TEXT


def test_followup_drops_thinking_and_returns_text():
    with _patch_api(_message()):
        assert answer_followup("Can I afford a car?") == PLAN_TEXT


def test_request_payload_suits_the_model():
    """Opus 5 rejects sampling params; thinking and effort replace them."""
    payload = build_llm()._get_request_payload([])

    assert payload["model"] == advisor_agent.MODEL
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": advisor_agent.EFFORT}
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload


# --- streaming ----------------------------------------------------------------

def test_stream_yields_only_text_chunks():
    with _patch_api(_stream_events()):
        assert list(stream_financial_advisor()) == [
            "### 1. Where ", "You Stand\n", "You save $1,800/mo.",
        ]


def test_stream_joins_back_to_the_full_plan():
    with _patch_api(_stream_events()):
        assert "".join(stream_financial_advisor()) == PLAN_TEXT


def test_stream_followup_yields_text():
    with _patch_api(_stream_events()):
        assert "".join(stream_followup("What about a car?")) == PLAN_TEXT


# --- credentials --------------------------------------------------------------

def test_missing_key_raises_before_any_network_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        run_financial_advisor()
