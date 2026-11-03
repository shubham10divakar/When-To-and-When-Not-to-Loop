"""OpenAI Chat Completions client (e.g. gpt-4.1-nano, gpt-5-nano).

* Streaming with ``stream_options={"include_usage": True}``; TTFT is the wall
  time to the first chunk carrying content or a tool-call fragment.
* Tool calling: Anthropic-shaped history is converted to OpenAI messages
  (assistant ``tool_calls`` with JSON-string arguments + ``role: tool`` with
  ``tool_call_id``); one call per step via ``parallel_tool_calls=False``.
* Prompt caching is *automatic* on OpenAI (prefixes >= 1024 tokens) and cannot
  be disabled, so the naive arm is also billed with cache discounts. Usage maps
  ``prompt_tokens_details.cached_tokens`` -> ``cache_read`` and the rest of the
  prompt -> ``input_new``; reasoning tokens are part of ``output``.
* ``temperature=0`` + ``seed`` for non-reasoning models; reasoning models
  (gpt-5*, o*) get ``reasoning_effort`` instead (they reject temperature).
* Our own backoff on 429/5xx with the waiting time recorded.
"""

from __future__ import annotations

import json
import os
import time

from loopbounds.harness.llm.base import ContextOverflow, LLMClient, LLMResponse, ToolCall, Usage, strip_private

DEFAULT_MODEL = "gpt-4.1-nano"


def _is_reasoning(model: str) -> bool:
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def to_openai_tools(tools):
    if not tools:
        return None
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}} for t in tools]


def to_openai_messages(system, messages):
    out = []
    if system:
        out.append({"role": "system", "content": system})
    for m in strip_private(messages):
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                     for b in content if b.get("type") == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            texts = []
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    c = c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []))
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": c})
                elif b.get("type") == "text":
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


class OpenAIClient(LLMClient):
    kind = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, *, seed: int = 0, reasoning_effort: str = "minimal",
                 max_attempts: int = 8, max_backoff_s: float = 60.0, timeout_s: float = 600.0):
        import openai

        self._openai = openai
        self.client = openai.OpenAI(max_retries=0, timeout=timeout_s)
        self.model = model
        self.model_id = model
        self.seed = seed
        self.reasoning_effort = reasoning_effort
        self.max_attempts = max_attempts
        self.max_backoff_s = max_backoff_s

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def bind(self, task) -> None:
        self.seed += 1

    def _kwargs(self, system, messages, tools, max_tokens):
        kw = {"model": self.model, "messages": to_openai_messages(system, messages), "stream": True,
              "stream_options": {"include_usage": True}, "max_completion_tokens": int(max_tokens),
              "seed": self.seed}
        ot = to_openai_tools(tools)
        if ot:
            kw["tools"] = ot
            kw["tool_choice"] = "auto"
            kw["parallel_tool_calls"] = False
        if _is_reasoning(self.model):
            kw["reasoning_effort"] = self.reasoning_effort
        else:
            kw["temperature"] = 0
        return kw

    def complete(self, system, messages, tools, *, cache=False, max_tokens=1024) -> LLMResponse:
        O = self._openai
        kw = self._kwargs(system, messages, tools, max_tokens)
        waited = 0.0
        for attempt in range(self.max_attempts):
            t0 = time.perf_counter()
            ttft = None
            text, calls, usage, model_id, finish = [], {}, None, self.model, None
            try:
                stream = self.client.chat.completions.create(**kw)
                for chunk in stream:
                    model_id = getattr(chunk, "model", None) or model_id
                    if chunk.usage is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    ch = chunk.choices[0]
                    finish = ch.finish_reason or finish
                    d = ch.delta
                    if d is None:
                        continue
                    if ttft is None and (d.content or d.tool_calls):
                        ttft = time.perf_counter() - t0
                    if d.content:
                        text.append(d.content)
                    for tc in d.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function is not None:
                            slot["name"] += tc.function.name or ""
                            slot["args"] += tc.function.arguments or ""
                total = time.perf_counter() - t0
                break
            except O.BadRequestError as exc:
                msg = str(exc).lower()
                if "context" in msg and ("length" in msg or "maximum" in msg):
                    raise ContextOverflow(str(exc)) from exc
                raise
            except (O.RateLimitError, O.InternalServerError, O.APIConnectionError, O.APITimeoutError) as exc:
                if isinstance(exc, O.RateLimitError) and "quota" in str(exc).lower():
                    raise RuntimeError(f"OpenAI quota exhausted: {exc}") from exc
                wait = min(self.max_backoff_s, 2.0 ** attempt)
                time.sleep(wait)
                waited += wait
        else:
            raise RuntimeError("OpenAI API: retries exhausted")

        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
        out = int(getattr(usage, "completion_tokens", 0) or 0)
        u = Usage(input_new=prompt - cached, cache_read=cached, cache_write=0, output=out)
        content, tcs = [], []
        joined = "".join(text)
        if joined:
            content.append({"type": "text", "text": joined})
        for i in sorted(calls):
            c = calls[i]
            try:
                args = json.loads(c["args"]) if c["args"] else {}
            except ValueError:
                args = {}
            tid = c["id"] or f"call_{int(t0 * 1e6)}_{i}"
            content.append({"type": "tool_use", "id": tid, "name": c["name"], "input": args})
            tcs.append(ToolCall(c["name"], args, tid))
        if not content:
            content = [{"type": "text", "text": ""}]
        return LLMResponse(content=content, tool_calls=tcs, text=joined, usage=u,
                           ttft_s=ttft if ttft is not None else total, total_s=total, model_id=model_id,
                           stop_reason="tool_use" if tcs else finish, rate_limit_wait_s=waited)

    def summarize(self, system, messages, *, max_tokens=2048, history_tokens=None) -> LLMResponse:
        from loopbounds.harness.llm.anthropic_client import render_transcript

        transcript = render_transcript(strip_private(messages))
        return self.complete(system, [{"role": "user", "content": transcript}], None, max_tokens=max_tokens)
