"""LLM clients: real Anthropic client and scripted/stochastic mocks."""

from __future__ import annotations

from loopbounds.harness.llm.base import ContextOverflow, LLMClient, LLMResponse, ToolCall, Usage
from loopbounds.harness.llm.mock import CoupledMock, MarkovMock, PerfectMock

MOCKS = {"perfect_mock": PerfectMock, "markov_mock": MarkovMock, "coupled_mock": CoupledMock}


def make_client(spec: dict) -> LLMClient:
    """Build a client from an experiment config ``client:`` block."""
    spec = dict(spec or {})
    kind = spec.pop("kind", "perfect_mock")
    if kind in MOCKS:
        return MOCKS[kind](**spec)
    if kind == "anthropic":
        from loopbounds.harness.llm.anthropic_client import AnthropicClient

        return AnthropicClient(**spec)
    if kind == "ollama":
        from loopbounds.harness.llm.ollama_client import OllamaClient

        return OllamaClient(**spec)
    if kind == "openai":
        from loopbounds.harness.llm.openai_client import OpenAIClient

        return OpenAIClient(**spec)
    raise ValueError(f"unknown client kind {kind!r}")


__all__ = ["ContextOverflow", "LLMClient", "LLMResponse", "ToolCall", "Usage", "PerfectMock", "MarkovMock",
           "CoupledMock", "make_client"]
