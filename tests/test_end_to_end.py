"""End-to-end tests over real HTTP, with no API key.

Everything else in the suite mocks at the client-object level, which cannot
catch a broken request body, a misparsed response, or streaming that never
yields. These run a stub OpenAI-compatible server on localhost and drive the
advisor through the real client, so the whole path is exercised: request
construction -> HTTP -> response parsing -> text extraction.

What they do not verify is the quality of a real model's answer - that is the
model's job, not the code's.
"""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import advisor_agent

PLAN = (
    "### 1. Where You Stand\nSurplus is $1,800/mo.\n\n"
    "### 2. Monthly Savings Goal\nTarget $1,500/mo.\n\n"
    "### 3. Investment Strategy\n60% equities / 40% bonds.\n"
)

# Requests the stub received, so tests can assert on what was actually sent.
RECEIVED = []


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)

        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in PLAN.split(" "):
                chunk = {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.get("model"),
                    "choices": [
                        {"index": 0, "delta": {"content": word + " "},
                         "finish_reason": None}
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b'data: {"id":"c","object":"chat.completion.chunk",'
                             b'"created":0,"model":"stub","choices":'
                             b'[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
            return

        payload = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model"),
            "choices": [
                {"index": 0,
                 "message": {"role": "assistant", "content": PLAN},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17,
                      "total_tokens": 59},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def stub_server():
    """A stub OpenAI-compatible server on a free port."""
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    yield f"http://127.0.0.1:{server.server_port}/v1"

    server.shutdown()


@pytest.fixture(autouse=True)
def point_at_stub(stub_server, monkeypatch):
    RECEIVED.clear()

    monkeypatch.setenv("LLM_PROVIDER", "compatible")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "stub-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", stub_server)
    monkeypatch.setenv("OPENAI_COMPATIBLE_MODEL", "stub-model")
    # The session proxy must not intercept a localhost call.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def test_plan_round_trips_over_http():
    plan = advisor_agent.run_financial_advisor()

    assert "Monthly Savings Goal" in plan
    assert "60% equities" in plan


def test_the_request_carries_the_profile():
    """The prompt the model receives must contain the real figures."""
    advisor_agent.run_financial_advisor()

    sent = json.dumps(RECEIVED[0]["messages"])
    assert "$70,000" in sent          # income
    assert "1,500" in sent            # housing line from the expense table
    assert "moderate" in sent         # risk tolerance
    assert "save for a house" in sent # goal


def test_temperature_is_sent_to_an_openai_style_provider():
    advisor_agent.run_financial_advisor()

    assert RECEIVED[0]["temperature"] == pytest.approx(0.4)
    assert RECEIVED[0]["model"] == "stub-model"


def test_streaming_yields_progressively_and_joins_to_the_plan():
    chunks = list(advisor_agent.stream_financial_advisor())

    assert len(chunks) > 1, "streaming produced a single blob"
    assert "Monthly Savings Goal" in "".join(chunks)
    assert RECEIVED[0]["stream"] is True


def test_followup_question_reaches_the_model_with_its_history():
    advisor_agent.answer_followup(
        "How long until I can afford the house?",
        history=[("user", "hi"), ("assistant", "hello")],
    )

    sent = json.dumps(RECEIVED[0]["messages"])
    assert "How long until I can afford the house?" in sent
    assert "User: hi" in sent
    assert "Advisor: hello" in sent


def test_streamed_followup_round_trips():
    answer = "".join(
        advisor_agent.stream_followup("What about retirement?")
    )

    assert "Investment Strategy" in answer
