"""Tests for the provider layer.

The point of these is parameter gating: providers disagree about which
parameters are legal, and sending the wrong one is a hard API error rather
than a degraded response.
"""

import os
import sys

import pytest

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
