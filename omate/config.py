"""
Configuration manager with interactive setup.

Priority: environment variables > .env file > interactive prompt > defaults.
Supports API key input for OpenAI, Anthropic, and Ollama backends.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def load_config() -> None:
    """Load .env if present. Silent if missing."""
    try:
        from dotenv import load_dotenv
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path, override=False)
        else:
            example = Path(".env.example")
            if example.exists():
                load_dotenv(example, override=False)
    except ImportError:
        pass


def get_or_prompt(key: str, label: str, default: str = "",
                  secret: bool = False) -> str:
    """
    Return env var if set, otherwise prompt interactively.
    In non-interactive mode (CI, pipes) returns default silently.
    Placeholder values like 'sk-your-key-here' are treated as unset.
    """
    val = os.environ.get(key, "").strip()
    _placeholders = {"sk-your-key-here", "sk-ant-your-key-here", ""}
    if val and val not in _placeholders:
        return val
    if not sys.stdin.isatty():
        return default
    try:
        if secret:
            import getpass
            entered = getpass.getpass(f"  {label} (Enter to skip): ").strip()
        else:
            entered = input(f"  {label} [{default}]: ").strip() or default
        if entered and entered not in _placeholders:
            os.environ[key] = entered
            return entered
    except (EOFError, KeyboardInterrupt):
        pass
    return default


def configure_llm_interactive() -> str:
    """
    Interactive LLM backend wizard shown at demo startup.
    Lets the user pick a backend and enter API keys once.
    Returns the selected backend name.

    Skipped automatically in non-interactive environments (CI, pipes).
    """
    if not sys.stdin.isatty():
        return os.getenv("LLM_BACKEND", "mock")

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.prompt import Prompt
    except ImportError:
        return os.getenv("LLM_BACKEND", "mock")

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]Omate POC — LLM Backend Setup[/bold cyan]\n\n"
        "  [green]mock[/green]       No API key. Deterministic test responses.\n"
        "  [yellow]openai[/yellow]     GPT-4o-mini  (requires OPENAI_API_KEY)\n"
        "  [yellow]anthropic[/yellow]  Claude Haiku (requires ANTHROPIC_API_KEY)\n"
        "  [yellow]ollama[/yellow]     Local Mistral (Ollama must be running)\n\n"
        "[dim]Press Enter to keep default shown in brackets.[/dim]",
        border_style="cyan",
    ))

    backend = Prompt.ask(
        "  Select backend",
        choices=["mock", "openai", "anthropic", "ollama"],
        default=os.getenv("LLM_BACKEND", "mock"),
    )
    os.environ["LLM_BACKEND"] = backend

    if backend == "openai":
        get_or_prompt("OPENAI_API_KEY", "OpenAI API key (sk-...)", secret=True)
        get_or_prompt("OPENAI_MODEL", "OpenAI model", "gpt-4o-mini")

    elif backend == "anthropic":
        get_or_prompt("ANTHROPIC_API_KEY", "Anthropic API key (sk-ant-...)",
                      secret=True)
        get_or_prompt("ANTHROPIC_MODEL", "Anthropic model",
                      "claude-haiku-4-5-20251001")

    elif backend == "ollama":
        get_or_prompt("OLLAMA_BASE_URL", "Ollama base URL",
                      "http://localhost:11434")
        get_or_prompt("OLLAMA_MODEL", "Ollama model", "mistral")

    console.print(
        f"  [green]✓[/green] Backend: [cyan]{backend}[/cyan]\n"
    )
    return backend
