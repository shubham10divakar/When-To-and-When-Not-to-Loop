"""LLM client interface (design doc §5.8).

Messages use the Anthropic Messages API shape (list of ``{"role", "content"}``
dicts with content blocks). Keys starting with ``_`` (e.g. ``_ntok``, the
harness's own token annotation) are private: real clients strip them before
sending; mock clients read them to produce synthetic usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ContextOverflow(Exception):
    """The request exceeded the model's (or the harness's artificial) context window."""


@dataclass
class Usage:
    input_new: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0

    @property
    def context(self) -> int:
        return self.input_new + self.cache_read + self.cache_write

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(self.input_new + o.input_new, self.cache_read + o.cache_read,
                     self.cache_write + o.cache_write, self.output + o.output)


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]
    id: str


@dataclass
class LLMResponse:
    content: list[dict]                 # assistant content blocks, ready to append to history
    tool_calls: list[ToolCall]
    text: str
    usage: Usage
    ttft_s: float
    total_s: float
    model_id: str
    stop_reason: str | None = None
    rate_limit_wait_s: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def tool_call(self) -> ToolCall | None:
        return self.tool_calls[0] if self.tool_calls else None


def strip_private(obj):
    """Recursively drop keys starting with '_' (harness annotations)."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


class LLMClient(ABC):
    model_id: str = "unknown"
    kind: str = "base"

    def bind(self, task: Any) -> None:  # noqa: B027 - optional hook for mocks
        """Called at the start of each run with the task instance (mocks use it)."""

    @abstractmethod
    def complete(self, system: str | None, messages: list[dict], tools: list[dict] | None, *,
                 cache: bool = False, max_tokens: int = 1024) -> LLMResponse:
        ...

    def summarize(self, system: str, messages: list[dict], *, max_tokens: int = 2048,
                  history_tokens: int | None = None) -> LLMResponse:
        """Summarization call used by the summarized arm (default: a plain completion)."""
        return self.complete(system, messages, None, cache=False, max_tokens=max_tokens)
