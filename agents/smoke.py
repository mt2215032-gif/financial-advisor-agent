"""Make one real API call and show the result.

    python -m agents.smoke              # the provider from LLM_PROVIDER
    python -m agents.smoke openai       # a specific provider
    python -m agents.smoke --all        # every configured provider

Answers the only question the test suite cannot: does this actually reach a
model and get a reply back. Prints the reply, the latency, and token usage,
or the precise reason it failed.
"""

import sys
import time

from agents.providers import (
    PROVIDERS,
    MissingCredentialError,
    UnknownProviderError,
    base_url_for,
    build_chat_model,
    get_provider,
    model_name_for,
)

PROMPT = "Reply with exactly this word and nothing else: WORKING"


def _usage(response):
    """Token counts, when the provider reports them."""
    meta = getattr(response, "usage_metadata", None) or {}
    if not meta:
        return ""
    return (
        f"  tokens: {meta.get('input_tokens', '?')} in / "
        f"{meta.get('output_tokens', '?')} out"
    )


def _text(content):
    """Providers return either a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def check(provider_key):
    """Call one provider once. Returns True if it replied."""
    provider = PROVIDERS[provider_key]
    model = model_name_for(provider)
    endpoint = base_url_for(provider)

    print(f"\n[{provider.label}]")
    print(f"  model: {model}")
    if endpoint:
        print(f"  endpoint: {endpoint}")

    try:
        llm = build_chat_model(provider_key)
    except MissingCredentialError as err:
        print(f"  NOT CONFIGURED - {err}")
        return False
    except ImportError as err:
        print(f"  NOT INSTALLED - {err}")
        return False

    started = time.monotonic()
    try:
        response = llm.invoke(PROMPT)
    except Exception as err:
        print(f"  FAILED - {type(err).__name__}: {err}")
        if provider_key == "ollama":
            print("  (is `ollama serve` running, and has the model been pulled?)")
        return False

    elapsed = time.monotonic() - started
    reply = _text(response.content)

    print(f"  OK - replied in {elapsed:.1f}s")
    print(f"  reply: {reply!r}")
    usage = _usage(response)
    if usage:
        print(usage)
    return True


def main(argv):
    args = [a for a in argv if not a.startswith("-")]
    check_all = "--all" in argv

    if check_all:
        keys = list(PROVIDERS)
    elif args:
        keys = args
    else:
        try:
            keys = [get_provider().key]
        except UnknownProviderError as err:
            sys.exit(f"error: {err}")

    for key in keys:
        if key not in PROVIDERS:
            sys.exit(
                f"error: unknown provider '{key}'. "
                f"Choose one of: {', '.join(sorted(PROVIDERS))}"
            )

    print("Calling each provider once. This spends real tokens.")
    results = {key: check(key) for key in keys}

    working = [k for k, ok in results.items() if ok]
    print(f"\n{len(working)}/{len(results)} replied.", end=" ")

    if working:
        print(f"Run the app with:  LLM_PROVIDER={working[0]} streamlit run app.py")
        return 0

    print("Add a key to .env (see .env.example), or run a local model with Ollama.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
