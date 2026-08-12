try:
    import openai
except ImportError as e:
    print("Dependency missing. Please run `make install`.")
    exit(str(e))

import os
from modules import BasicLLM
from constants import OLLAMA_ENDPOINT, LMSTUDIO_ENDPOINT, full_config
from helpers import load_env

load_env()

# Determine provider safely (avoid KeyError when PROVIDER is not set)
endpoint = OLLAMA_ENDPOINT
provider = os.getenv("PROVIDER", "").strip().lower()
if not provider:
    print("Warning: PROVIDER not set. Defaulting to 'ollama'.")
    provider = "ollama"

if provider == "lmstudio":
    endpoint = LMSTUDIO_ENDPOINT
elif provider == "ollama":
    endpoint = OLLAMA_ENDPOINT
else:
    print(f"Warning: Unknown PROVIDER '{provider}'. Defaulting to 'ollama'.")
    endpoint = OLLAMA_ENDPOINT


def print_separator() -> None:
    print()
    print("==" * 20)


if __name__ == "__main__":
    # Use getenv for OPENAI_API so missing key doesn't raise KeyError
    api_key = os.getenv("OPENAI_API", "").strip()
    if not api_key:
        print("Error: OPENAI_API not set. If your provider requires an API key, set it in .env or the environment.")
        print("See .env.example for an example. Exiting to avoid interactive prompt when configuration is incomplete.")
        raise SystemExit(1)

    try:
        client = openai.OpenAI(base_url=endpoint, api_key=api_key)
    except Exception as e:
        print("Failed to initialize OpenAI client:", e)
        print("No API credentials found — falling back to a local dummy client for testing.")

        # Create a very small dummy client that matches the methods used by BasicLLM
        class _DummyCompletions:
            def create(self, **kwargs):
                class _ChoiceMessage:
                    def __init__(self):
                        self.message = type("m", (), {"content": "[no-api] No API key configured. Set OPENAI_API in .env to use a real model."})

                return type("R", (), {"choices": [_ChoiceMessage()]})

        class _DummyChat:
            def __init__(self):
                self.completions = _DummyCompletions()

        class _DummyClient:
            def __init__(self):
                self.chat = _DummyChat()

        client = _DummyClient()
        # Ensure non-streaming mode for the dummy client
        try:
            full_config["stream"] = False
        except Exception:
            pass

        print("Exiting because no valid API credentials were found.\n" \
              "To run interactively, set `OPENAI_API` and `PROVIDER` in a .env file or in the environment.\n" \
              "See .env.example for an example.")
        raise SystemExit(0)
    llm = BasicLLM(client)
    while True:
        print_separator()

        prompt = input("User> ")

        print_separator()

        if prompt.lower() == "exit":
            break

        llm.run(command=prompt, **full_config)
