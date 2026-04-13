"""
LLM backend adapters.

Supports four backends:
  - "mock"      : deterministic fake responses (no API key, for testing)
  - "openai"    : OpenAI API (gpt-4o-mini by default)
  - "anthropic" : Anthropic API (claude-haiku-4-5-20251001 by default)
  - "ollama"    : local Ollama server (mistral by default)

Switch via LLM_BACKEND environment variable.
API keys are read from environment variables or set interactively via
omate.config.configure_llm_interactive().
"""

import os
import random
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    model: str
    temperature: float


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        pass


class MockLLM(LLMBackend):
    """
    Deterministic mock LLM — no API key required.

    Returns a structured clinical report template populated with
    data extracted from the prompt context.
    Useful for testing the full pipeline without any LLM dependency.
    """

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        # Extract patient ID from prompt if present
        patient_id = "unknown"
        for line in prompt.split("\n"):
            if "patient_id" in line.lower() or "patient/" in line.lower():
                parts = line.split(":")
                if len(parts) > 1:
                    patient_id = parts[-1].strip().strip('"').split("/")[-1]
                    break

        # Vary slightly with temperature for SelfCheckGPT testing
        variant = random.choice([
            "within expected clinical parameters",
            "consistent with recent monitoring data",
            "showing stable trends",
        ]) if temperature > 0 else "within expected clinical parameters"

        text = textwrap.dedent(f"""
            Clinical Summary — Patient {patient_id}

            Based on the available monitoring data, the patient's cardiac
            rhythm is {variant} [SOURCE: resource_id=obs-demo-1].

            ECG monitoring over the past 24 hours shows recorded observations
            stored in the system [SOURCE: resource_id=obs-demo-2]. Heart rate
            and rhythm parameters have been logged for physician review
            [SOURCE: resource_id=obs-demo-3].

            Glucose levels have been within the monitored range based on
            available CGM readings [SOURCE: resource_id=obs-demo-4].

            Recommendation: Physician review of all flagged observations
            is required before finalizing this report.
        """).strip()

        return LLMResponse(text=text, model="mock-v1", temperature=temperature)


class OpenAILLM(LLMBackend):
    """OpenAI API backend."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=800,
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            temperature=temperature,
        )


class AnthropicLLM(LLMBackend):
    """
    Anthropic API backend.

    Uses the Messages API. Requires ANTHROPIC_API_KEY.
    Default model: claude-haiku-4-5-20251001 (fast, cheap, good at structured tasks).
    For higher accuracy swap to claude-sonnet-4-6 or claude-opus-4-6.

    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    export LLM_BACKEND=anthropic
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
                 api_key: str | None = None):
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError:
            raise ImportError(
                "pip install anthropic  # then set ANTHROPIC_API_KEY"
            )
        self.client = self._anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=message.content[0].text,
            model=self.model,
            temperature=temperature,
        )


class OllamaLLM(LLMBackend):
    """Local Ollama backend."""

    def __init__(self, model: str = "mistral",
                  base_url: str = "http://localhost:11434"):
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("pip install requests")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.0) -> LLMResponse:
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt,
                  "options": {"temperature": temperature}, "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        return LLMResponse(text=text, model=self.model, temperature=temperature)


def get_llm_backend() -> LLMBackend:
    """
    Factory — returns the backend specified by LLM_BACKEND env var.
    Defaults to MockLLM if not set or unrecognised.

    Supported values for LLM_BACKEND:
      mock      — no API key needed (default)
      openai    — requires OPENAI_API_KEY
      anthropic — requires ANTHROPIC_API_KEY
      ollama    — requires Ollama server running locally
    """
    backend = os.getenv("LLM_BACKEND", "mock").lower()
    if backend == "openai":
        return OpenAILLM(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif backend == "anthropic":
        return AnthropicLLM(
            model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    elif backend == "ollama":
        return OllamaLLM(
            model=os.getenv("OLLAMA_MODEL", "mistral"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        return MockLLM()
