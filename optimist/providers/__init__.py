"""LLM provider backends."""

from __future__ import annotations

from optimist.providers.base import Provider, StreamHandler

__all__ = ["Provider", "StreamHandler", "get_provider"]

_CACHE: dict[str, Provider] = {}


def get_provider(name: str) -> Provider:
    """Return a (cached) provider instance by name: 'anthropic' or 'openrouter'."""
    if name in _CACHE:
        return _CACHE[name]
    if name == "anthropic":
        from optimist.providers.anthropic_provider import AnthropicProvider

        _CACHE[name] = AnthropicProvider()
    elif name == "openrouter":
        from optimist.providers.openrouter_provider import OpenRouterProvider

        _CACHE[name] = OpenRouterProvider()
    else:
        raise ValueError(f"Unknown provider: {name!r}")
    return _CACHE[name]
