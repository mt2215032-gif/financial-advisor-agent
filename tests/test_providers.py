"""Tests for the provider layer.

The point of these is parameter gating: providers disagree about which
parameters are legal, and sending the wrong one is a hard API error rather
than a degraded response.
"""

import os
import sys

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.providers import (
    PROVIDERS,
    MissingCredentialError,
    UnknownProviderError,
    base_url_for,
    build_chat_model,
    get_provider,
    model_name_for,
)
import advisor_agent
from agents.settings import MissingAPIKeyError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start each test from a known environment."""
    for var in (
        "LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_MODEL", "OPENAI_MODEL",
        "OLLAMA_MODEL", "OPENAI_COMPATIBLE_MODEL", "OPENAI_COMPATIBLE_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "sk-compat-test")
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "https://api.example.com/v1")


# --- selection ----------------------------------------------------------------

def test_defaults_to_claude():
    assert get_provider().key == "anthropic"


def test_env_var_selects_the_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert get_provider().key == "openai"


def test_selection_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")
    assert get_provider().key == "openai"


def test_unknown_provider_lists_the_valid_ones():
    with pytest.raises(UnknownProviderError, match="anthropic"):
        get_provider("gemini")


def test_llm_model_overrides_every_provider(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "custom-model")

    for key in PROVIDERS:
        assert model_name_for(PROVIDERS[key]) == "custom-model"


def test_per_provider_model_var_is_used(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    assert model_name_for(PROVIDERS["openai"]) == "gpt-4o-mini"
    # ...and does not leak into another provider.
    assert model_name_for(PROVIDERS["anthropic"]) == "claude-opus-5"


def test_ollama_has_a_local_default_base_url():
    assert base_url_for(PROVIDERS["ollama"]) == "http://localhost:11434"


# --- parameter gating ---------------------------------------------------------

def test_temperature_never_reaches_claude():
    """Opus 5 returns 400 for temperature/top_p/top_k."""
    payload = build_chat_model("anthropic")._get_request_payload([])

    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "top_k" not in payload


def test_claude_gets_adaptive_thinking_instead():
    payload = build_chat_model("anthropic")._get_request_payload([])

    assert payload["thinking"] == {"type": "adaptive"}
    assert "effort" in payload["output_config"]


@pytest.mark.parametrize("key", ["openai", "ollama", "compatible"])
def test_other_providers_do_get_temperature(key):
    assert build_chat_model(key).temperature is not None


@pytest.mark.parametrize("key", ["openai", "ollama", "compatible"])
def test_thinking_config_is_not_sent_to_non_anthropic(key):
    model = build_chat_model(key)

    assert not hasattr(model, "thinking") or model.thinking is None


# --- construction -------------------------------------------------------------

def test_every_provider_builds():
    expected = {
        "anthropic": "ChatAnthropic",
        "openai": "ChatOpenAI",
        "ollama": "ChatOllama",
        "compatible": "ChatOpenAI",
    }

    for key, class_name in expected.items():
        assert type(build_chat_model(key)).__name__ == class_name


def test_compatible_endpoint_routes_to_its_base_url():
    model = build_chat_model("compatible")
    assert "api.example.com" in str(model.openai_api_base)


def test_ollama_needs_no_api_key(monkeypatch):
    """Local models must work with no credentials configured at all."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert build_chat_model("ollama") is not None


# --- credentials --------------------------------------------------------------

def test_missing_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY"):
        build_chat_model("openai")


def test_compatible_endpoint_requires_a_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)

    with pytest.raises(MissingCredentialError, match="OPENAI_COMPATIBLE_BASE_URL"):
        build_chat_model("compatible")


def test_credential_error_is_catchable_as_the_original_type(monkeypatch):
    """Existing `except MissingAPIKeyError` handlers must still catch these."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        build_chat_model("openai")


# --- provider-agnostic response handling --------------------------------------
# Claude returns a list of content blocks (thinking, text, ...); OpenAI and
# Ollama return a plain string. The advisor path must handle both.

class _FakeModel:
    """Stands in for any LangChain chat model, with configurable content."""

    def __init__(self, content, chunks=None):
        self._content = content
        self._chunks = chunks or []

    def invoke(self, messages):
        return AIMessage(content=self._content)

    def stream(self, messages):
        for chunk in self._chunks:
            yield AIMessageChunk(content=chunk)


def _patch_model(monkeypatch, model):
    monkeypatch.setattr(advisor_agent, "build_chat_model",
                        lambda **kwargs: model)


def test_plain_string_response_is_returned_as_is(monkeypatch):
    """OpenAI and Ollama shape: content is already a string."""
    _patch_model(monkeypatch, _FakeModel("Save $1,800/mo."))

    assert advisor_agent.run_financial_advisor() == "Save $1,800/mo."


def test_block_list_response_drops_thinking(monkeypatch):
    """Claude shape: a list of blocks, only the text ones count."""
    _patch_model(monkeypatch, _FakeModel([
        {"type": "thinking", "thinking": "internal"},
        {"type": "text", "text": "Save $1,800/mo."},
    ]))

    assert advisor_agent.run_financial_advisor() == "Save $1,800/mo."


def test_streaming_plain_string_chunks(monkeypatch):
    """Non-Anthropic providers stream bare strings, not blocks."""
    _patch_model(monkeypatch, _FakeModel("", chunks=["Save ", "$1,800", "/mo."]))

    assert "".join(advisor_agent.stream_financial_advisor()) == "Save $1,800/mo."


def test_streaming_block_chunks(monkeypatch):
    """Anthropic streams blocks; empty thinking deltas must not appear."""
    _patch_model(monkeypatch, _FakeModel("", chunks=[
        [{"type": "thinking", "thinking": "..."}],
        [{"type": "text", "text": "Save "}],
        [{"type": "text", "text": "$1,800/mo."}],
    ]))

    assert "".join(advisor_agent.stream_financial_advisor()) == "Save $1,800/mo."


def test_followup_works_on_a_plain_string_provider(monkeypatch):
    _patch_model(monkeypatch, _FakeModel("Yes, in about 31 months."))

    answer = advisor_agent.answer_followup(
        "How long for the house?", history=[("user", "hi")]
    )
    assert answer == "Yes, in about 31 months."


# --- import-time resilience ---------------------------------------------------

def test_model_attribute_follows_the_active_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert advisor_agent.MODEL == "gpt-4o"

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert advisor_agent.MODEL == "claude-opus-5"


def test_bad_provider_does_not_break_import(monkeypatch):
    """A typo in LLM_PROVIDER must not crash the app before it can render;
    the error surfaces on use, where the UI can catch and display it."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    import importlib
    importlib.reload(advisor_agent)          # no raise

    with pytest.raises(UnknownProviderError):
        advisor_agent.MODEL


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        advisor_agent.not_a_real_attribute
