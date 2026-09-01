"""Provider layer: run the advisor on Claude, OpenAI, or open-source models.

Every LangChain-based agent in this project builds its model here, so swapping
providers is one environment variable and no code change:

    LLM_PROVIDER=anthropic   # Claude (default)
    LLM_PROVIDER=openai      # OpenAI API
    LLM_PROVIDER=ollama      # local open-source models (Llama, Mistral, Qwen)
    LLM_PROVIDER=compatible  # any OpenAI-compatible endpoint

The providers are not interchangeable at the parameter level, which is the
whole reason this file exists:

- Claude Opus 5 **rejects** `temperature` / `top_p` / `top_k` with a 400, and
  takes adaptive thinking plus an effort level instead.
- OpenAI and Ollama take `temperature` and have no equivalent of that
  thinking config.

So each provider declares what it accepts, and only those parameters are sent.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from agents.settings import MissingAPIKeyError

load_dotenv()

DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
MAX_TOKENS = 16000


@dataclass(frozen=True)
class Provider:
    """What one provider is called, needs, and accepts."""

    key: str
    label: str
    default_model: str
    env_key: str | None          # None when no API key applies (local models)
    package: str                 # pip package supplying the integration
    supports_temperature: bool
    supports_thinking: bool      # Anthropic-style adaptive thinking + effort
    base_url_env: str | None = None
    default_base_url: str | None = None


PROVIDERS = {
    "anthropic": Provider(
        key="anthropic",
        label="Claude (Anthropic)",
        default_model="claude-opus-5",
        env_key="ANTHROPIC_API_KEY",
        package="langchain-anthropic",
        supports_temperature=False,   # Opus 5 returns 400 for temperature
        supports_thinking=True,
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI",
        default_model="gpt-4o",
        env_key="OPENAI_API_KEY",
        package="langchain-openai",
        supports_temperature=True,
        supports_thinking=False,
    ),
    "ollama": Provider(
        key="ollama",
        label="Ollama (local open-source)",
        default_model="llama3.1",
        env_key=None,                 # runs on your machine, no key
        package="langchain-ollama",
        supports_temperature=True,
        supports_thinking=False,
        base_url_env="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434",
    ),
    "compatible": Provider(
        key="compatible",
        label="OpenAI-compatible endpoint",
        default_model="meta-llama/Llama-3.1-70B-Instruct",
        env_key="OPENAI_COMPATIBLE_API_KEY",
        package="langchain-openai",
        supports_temperature=True,
        supports_thinking=False,
        base_url_env="OPENAI_COMPATIBLE_BASE_URL",
    ),
}

DEFAULT_PROVIDER = "anthropic"


class UnknownProviderError(ValueError):
    """Raised for an LLM_PROVIDER value that is not in PROVIDERS."""


class MissingCredentialError(MissingAPIKeyError):
    """Raised when the selected provider's API key is not set.

    Subclasses MissingAPIKeyError so callers that already handle the
    Anthropic-only case keep working for every provider.
    """


def get_provider(name=None):
    """Resolve the provider, from the argument or LLM_PROVIDER."""
    name = (name or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()

    if name not in PROVIDERS:
        raise UnknownProviderError(
            f"Unknown LLM_PROVIDER '{name}'. Choose one of: "
            f"{', '.join(sorted(PROVIDERS))}"
        )
    return PROVIDERS[name]


def model_name_for(provider):
    """The model id, from a provider-specific env var or the default.

    LLM_MODEL overrides for whichever provider is selected; the per-provider
    variables let you keep several configured at once.
    """
    per_provider = {
        "anthropic": "ANTHROPIC_MODEL",
        "openai": "OPENAI_MODEL",
        "ollama": "OLLAMA_MODEL",
        "compatible": "OPENAI_COMPATIBLE_MODEL",
    }[provider.key]

    return (
        os.getenv("LLM_MODEL")
        or os.getenv(per_provider)
        or provider.default_model
    )


def base_url_for(provider):
    """The endpoint URL, where the provider needs one."""
    if not provider.base_url_env:
        return None
    return os.getenv(provider.base_url_env) or provider.default_base_url


def require_credentials(provider):
    """Check the provider's key up front, with a message naming the fix."""
    if provider.env_key is None:
        return None

    value = os.getenv(provider.env_key)
    if not value:
        raise MissingCredentialError(
            f"{provider.env_key} is not set, and provider '{provider.key}' "
            f"({provider.label}) needs it. Copy .env.example to .env and add "
            f"the key, or set LLM_PROVIDER to a different provider."
        )

    if provider.base_url_env and not base_url_for(provider):
        raise MissingCredentialError(
            f"{provider.base_url_env} is not set. Point it at your "
            f"OpenAI-compatible endpoint, e.g. https://api.groq.com/openai/v1"
        )

    return value


def _import_error(provider, err):
    return ImportError(
        f"Missing dependency '{err.name}' for provider '{provider.key}' "
        f"({provider.label}). Install it with:\n"
        f"    pip install {provider.package}"
    )


def build_chat_model(provider=None, streaming=False, **overrides):
    """Build the LangChain chat model for the selected provider.

    Only parameters the provider actually accepts are passed - sending
    `temperature` to Claude Opus 5 is a 400, and sending `thinking` to OpenAI
    is meaningless.
    """
    provider = provider if isinstance(provider, Provider) else get_provider(provider)
    api_key = require_credentials(provider)
    model = overrides.pop("model", None) or model_name_for(provider)

    if provider.key == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as err:
            raise _import_error(provider, err) from err

        # No temperature: adaptive thinking + effort steer depth instead.
        return ChatAnthropic(
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": os.getenv("ANTHROPIC_EFFORT", "medium")},
            streaming=streaming,
            **overrides,
        )

    if provider.key == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as err:
            raise _import_error(provider, err) from err

        # Ollama caps output with num_predict rather than max_tokens.
        return ChatOllama(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            base_url=base_url_for(provider),
            num_predict=MAX_TOKENS,
            **overrides,
        )

    # openai and compatible share the OpenAI client; only the endpoint differs.
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as err:
        raise _import_error(provider, err) from err

    kwargs = {
        "model": model,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "streaming": streaming,
        "api_key": api_key,
    }
    if base_url_for(provider):
        kwargs["base_url"] = base_url_for(provider)

    return ChatOpenAI(**kwargs, **overrides)


def describe_active():
    """A one-line description of the current selection, for the UI."""
    provider = get_provider()
    return f"{provider.label} · {model_name_for(provider)}"
