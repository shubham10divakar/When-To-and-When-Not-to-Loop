"""Real Anthropic Messages API client (design doc §5.8).

* Streaming, so time-to-first-token can be measured (first content delta).
* Cached arm: a single ``cache_control`` breakpoint on the last content block of
  the last message (incremental prefix caching; tools → system → messages render
  order means tools + system are cached with it). Naive arm: no markers.
* Our own exponential backoff on 429 / 5xx / connection errors; the waiting
  time is recorded and excluded from latency analysis. SDK retries are off so
  every wait is visible.
* ``model_id`` is the exact id returned by the API on every response.

Notes for current models: sampling parameters (``temperature`` etc.) are
rejected by the API, so determinism is controlled with a fixed ``effort`` and
thinking configuration instead of temperature 0. Forced ``tool_choice`` is not
used; ``auto`` with parallel tool use disabled gives one call per step.
"""

from __future__ import annotations

import copy
import json
import os
import time

from loopbounds.harness.llm.base import ContextOverflow, LLMClient, LLMResponse, ToolCall, Usage, strip_private

DEFAULT_MODEL = "claude-haiku-5-5"


def _block_to_dict(b) -> dict:
    d = b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else dict(b)
    return d


def _with_breakpoint(messages: list[dict]) -> list[dict]:
    msgs = copy.deepcopy(messages)
    for m in msgs:                       # at most one breakpoint: remove stale markers
        if isinstance(m["content"], list):
            for blk in m["content"]:
                blk.pop("cache_control", None)
    last = msgs[-1]
    if isinstance(last["content"], str):
        last["content"] = [{"type": "text", "text": last["content"]}]
    last["content"][-1]["cache_control"] = {"type": "ephemeral"}
    return msgs


def render_transcript(messages: list[dict]) -> str:
    """Plain-text transcript of a tool-use history (used by the summarization call)."""
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append(f"{m['role'].upper()}: {content}")
            continue
        for blk in content:
            t = blk.get("type")
            if t == "text":
                out.append(f"{m['role'].upper()}: {blk['text']}")
            elif t == "tool_use":
                out.append(f"ASSISTANT called {blk['name']}({json.dumps(blk.get('input', {}))})")
            elif t == "tool_result":
                c = blk.get("content")
                c = c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []))
                out.append(f"TOOL RESULT:\n{c}")
    return "\n\n".join(out)


class AnthropicClient(LLMClient):
    kind = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, *, effort: str | None = "low",
                 thinking: str | None = "adaptive", max_backoff_s: float = 60.0, max_attempts: int = 8,
                 timeout_s: float = 600.0):
        import anthropic  # imported lazily so the offline pipeline never needs it

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=0, timeout=timeout_s)
        self.model = model
        self.model_id = model
        self.effort = effort
        self.thinking = thinking
        self.max_backoff_s = max_backoff_s
        self.max_attempts = max_attempts

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))

    def _kwargs(self, system, messages, tools, cache, max_tokens) -> dict:
        msgs = strip_private(messages)
        if cache:
            msgs = _with_breakpoint(msgs)
        kw: dict = {"model": self.model, "max_tokens": max_tokens, "messages": msgs}
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if self.thinking:
            kw["thinking"] = {"type": self.thinking}
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        return kw

    def complete(self, system, messages, tools, *, cache=False, max_tokens=1024) -> LLMResponse:
        A = self._anthropic
        kw = self._kwargs(system, messages, tools, cache, max_tokens)
        waited = 0.0
        for attempt in range(self.max_attempts):
            t0 = time.perf_counter()
            ttft = None
            try:
                with self.client.messages.stream(**kw) as stream:
                    for event in stream:
                        if ttft is None and event.type in ("content_block_delta", "content_block_start"):
                            ttft = time.perf_counter() - t0
                    msg = stream.get_final_message()
                total = time.perf_counter() - t0
                break
            except A.BadRequestError as exc:
                text = str(exc).lower()
                if "too long" in text or ("context" in text and "exceed" in text):
                    raise ContextOverflow(str(exc)) from exc
                raise
            except (A.RateLimitError, A.InternalServerError, A.APIConnectionError) as exc:
                retry_after = None
                resp = getattr(exc, "response", None)
                if resp is not None:
                    try:
                        retry_after = float(resp.headers.get("retry-after"))
                    except (TypeError, ValueError):
                        retry_after = None
                wait = retry_after if retry_after is not None else min(self.max_backoff_s, 2.0 ** attempt)
                time.sleep(wait)
                waited += wait
            except A.APIStatusError as exc:
                if getattr(exc, "status_code", 0) >= 500:
                    wait = min(self.max_backoff_s, 2.0 ** attempt)
                    time.sleep(wait)
                    waited += wait
                    continue
                raise
        else:
            raise RuntimeError("Anthropic API: retries exhausted")

        u = msg.usage
        usage = Usage(input_new=int(u.input_tokens or 0),
                      cache_read=int(getattr(u, "cache_read_input_tokens", 0) or 0),
                      cache_write=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                      output=int(u.output_tokens or 0))
        content = [_block_to_dict(b) for b in msg.content]
        calls = [ToolCall(b["name"], dict(b.get("input") or {}), b["id"]) for b in content if b.get("type") == "tool_use"]
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        return LLMResponse(content=content, tool_calls=calls, text=text, usage=usage,
                           ttft_s=ttft if ttft is not None else total, total_s=total, model_id=msg.model,
                           stop_reason=msg.stop_reason, rate_limit_wait_s=waited)

    def summarize(self, system, messages, *, max_tokens=2048, history_tokens=None) -> LLMResponse:
        transcript = render_transcript(strip_private(messages))
        return self.complete(system, [{"role": "user", "content": transcript}], None, cache=False,
                             max_tokens=max_tokens)
