"""Settings: model selection, provider inference, limits.

Loaded from (in priority order) environment variables, an optional TOML file
at ~/.config/optimist/config.toml, and defaults.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5"
CONFIG_PATH = Path(os.environ.get("OPTIMIST_CONFIG", "~/.config/optimist/config.toml"))


def provider_for(model: str) -> str:
    """Infer the provider from a model id.

    claude-* ids go to Anthropic; explicit 'openrouter:' / 'or:' prefixes and
    vendor-scoped ids like 'deepseek/deepseek-v4' go to OpenRouter.
    """
    if model.startswith(("openrouter:", "or:")):
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if "/" in model:
        return "openrouter"
    return "anthropic"


def strip_model_prefix(model: str) -> str:
    for prefix in ("openrouter:", "or:", "anthropic:"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


@dataclass
class Settings:
    model: str = DEFAULT_MODEL
    # Sub-agents (formulator/solver/verifier) can run a different model;
    # empty string means "same as the main model".
    subagent_model: str = ""
    max_tokens: int = 32000
    sandbox_timeout_s: int = 120
    show_thinking: bool = False
    history_file: Path = field(
        default_factory=lambda: Path("~/.local/share/optimist/history").expanduser()
    )

    @classmethod
    def load(cls) -> Settings:
        s = cls()
        path = CONFIG_PATH.expanduser()
        if path.is_file():
            try:
                data = tomllib.loads(path.read_text())
            except (OSError, tomllib.TOMLDecodeError):
                data = {}
            for key in ("model", "subagent_model"):
                if isinstance(data.get(key), str):
                    setattr(s, key, data[key])
            for key in ("max_tokens", "sandbox_timeout_s"):
                if isinstance(data.get(key), int):
                    setattr(s, key, data[key])
            if isinstance(data.get("show_thinking"), bool):
                s.show_thinking = data["show_thinking"]
        if os.environ.get("OPTIMIST_MODEL"):
            s.model = os.environ["OPTIMIST_MODEL"]
        if os.environ.get("OPTIMIST_SUBAGENT_MODEL"):
            s.subagent_model = os.environ["OPTIMIST_SUBAGENT_MODEL"]
        return s

    def effective_subagent_model(self) -> str:
        return self.subagent_model or self.model
