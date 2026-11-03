"""Local open-source models through Ollama's native chat API (``/api/chat``).

* Tool calling: our Anthropic-shaped history (tool_use / tool_result blocks) is
  converted to Ollama's message format (assistant ``tool_calls`` + ``role: tool``).
* Streaming: TTFT is the wall time to the first streamed chunk that carries
  content or a tool call. Ollama's server-side timings (``prompt_eval_duration``,
  ``eval_duration``, ``load_duration``) are kept in ``extra``.
* Context window: Ollama silently truncates prompts longer than ``num_ctx``, so
  ``num_ctx`` is always set explicitly and the runner should use a
  ``window_cap`` below it (see :data:`WINDOW_SAFETY`).
* Token counts come from Ollama's real tokenizer (``prompt_eval_count``,
  ``eval_count``). Ollama may reuse a cached KV prefix on its own; if it then
  reports far fewer prompt tokens than the previous context, the difference is
  logged as ``cache_read``. Cost is $0 for local models; the relevant wall is
  latency, and token counts still give the hosted-API cost of the same run.
* Determinism: ``temperature`` 0 and a fixed ``seed``.
"""

from __future__ import annotations

import json
import time

import httpx

from loopbounds.harness.llm.base import ContextOverflow, LLMClient, LLMResponse, ToolCall, Usage, strip_private
from loopbounds.harness.llm.mock import MockClient
from loopbounds.harness.tokens import count_tokens

DEFAULT_HOST = "http://localhost:11434"
WINDOW_SAFETY = 0.85     # runner window_cap = WINDOW_SAFETY * num_ctx (token estimate vs real tokenizer)


def to_ollama_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}} for t in tools]


def to_ollama_messages(system: str | None, messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    names: dict[str, str] = {}
    for m in strip_private(messages):
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = []
            for b in content:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append({"function": {"name": b["name"], "arguments": b.get("input") or {}}})
            msg = {"role": "assistant", "content": text}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            texts = []
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    c = c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []))
                    out.append({"role": "tool", "content": c, "tool_name": names.get(b.get("tool_use_id"), "")})
                elif b.get("type") == "text":
                    texts.append(b["text"])
            if texts:
                out.append({"role": "user", "content": "\n".join(texts)})
    return out


_TOOL_TEXT_MARKERS = ("[TOOL_CALLS]", "<tool_call>", "<|python_tag|>")


def parse_text_tool_calls(text: str, tool_names: set[str]) -> list[dict]:
    """Recover tool calls a model wrote as text instead of structured ``tool_calls``.

    Some models (e.g. mistral:7b) sometimes emit ``[TOOL_CALLS] [{"name": ..., "arguments": ...}]``
    as plain content, which Ollama's parser misses. Only JSON objects whose ``name`` is a known
    tool are accepted. Returns Ollama-shaped ``{"function": {"name", "arguments"}}`` dicts.
    """
    if not text or not tool_names:
        return []
    s = text
    for mk in _TOOL_TEXT_MARKERS:
        s = s.replace(mk, " ")
    s = s.replace("</tool_call>", " ")
    dec = json.JSONDecoder()
    out: list[dict] = []
    i = 0
    while i < len(s):
        if s[i] not in "[{":
            i += 1
            continue
        try:
            v, j = dec.raw_decode(s, i)
        except ValueError:
            i += 1
            continue
        for obj in (v if isinstance(v, list) else [v]):
            if isinstance(obj, dict):
                fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
                name = fn.get("name")
                if name in tool_names:
                    args = fn.get("arguments", fn.get("parameters", {}))
                    out.append({"function": {"name": name, "arguments": args}})
        i = j
    return out


def _parse_args(a) -> dict:
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a)
        return v if isinstance(v, dict) else {}
    except (TypeError, ValueError):
        return {}


class OllamaClient(LLMClient):
    kind = "ollama"

    def __init__(self, model: str = "llama3.1:8b", *, host: str = DEFAULT_HOST, num_ctx: int = 32768,
                 temperature: float = 0.0, seed: int = 0, timeout_s: float = 1800.0, keep_alive: str = "30m"):
        self.model = model
        self.model_id = f"ollama/{model}"
        self.host = host.rstrip("/")
        self.num_ctx = int(num_ctx)
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": self.num_ctx}
        self.keep_alive = keep_alive
        self.http = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self._seq_seed = seed
        self._prev_prompt = 0

    @property
    def window_cap(self) -> int:
        return int(WINDOW_SAFETY * self.num_ctx)

    def available(self) -> bool:
        try:
            tags = self.http.get(f"{self.host}/api/tags").json()
            return any(m.get("name") == self.model or m.get("model") == self.model for m in tags.get("models", []))
        except httpx.HTTPError:
            return False

    def bind(self, task) -> None:
        # a different but reproducible sampling seed per run
        self._seq_seed += 1
        self.options["seed"] = self._seq_seed
        self._prev_prompt = 0

    def complete(self, system, messages, tools, *, cache=False, max_tokens=1024) -> LLMResponse:
        body = {"model": self.model, "messages": to_ollama_messages(system, messages), "stream": True,
                "options": {**self.options, "num_predict": int(max_tokens)}, "keep_alive": self.keep_alive}
        ot = to_ollama_tools(tools)
        if ot:
            body["tools"] = ot
        est_total = MockClient.prompt_tokens(system, messages, tools)
        if est_total > self.num_ctx:
            raise ContextOverflow(f"estimated prompt {est_total} tokens > num_ctx {self.num_ctx}")
        t0 = time.perf_counter()
        ttft = None
        text_parts: list[str] = []
        calls: list[dict] = []
        final: dict = {}
        model_error: str | None = None
        with self.http.stream("POST", f"{self.host}/api/chat", json=body) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"ollama {resp.status_code}: {resp.read().decode(errors='replace')[:500]}")
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    # generation-time failure (e.g. "token repeat limit reached"): the model
                    # produced degenerate output. Treat it as a turn without a tool call.
                    model_error = str(chunk["error"])
                    break
                msg = chunk.get("message") or {}
                if ttft is None and (msg.get("content") or msg.get("tool_calls") or msg.get("thinking")):
                    ttft = time.perf_counter() - t0
                if msg.get("content"):
                    text_parts.append(msg["content"])
                if msg.get("tool_calls"):
                    calls.extend(msg["tool_calls"])
                if chunk.get("done"):
                    final = chunk
        total = time.perf_counter() - t0
        new = int(final.get("prompt_eval_count") or 0)
        out = int(final.get("eval_count") or 0)
        # Ollama normally reports the full prompt (real tokenizer). If it reports far fewer
        # tokens than the prompt holds, the prefix came from its KV cache.
        if self._prev_prompt and new < 0.5 * self._prev_prompt:
            cached = self._prev_prompt
            usage = Usage(input_new=new, cache_read=cached, cache_write=0, output=out)
            self._prev_prompt = cached + new + out
        else:
            usage = Usage(input_new=new, cache_read=0, cache_write=0, output=out)
            self._prev_prompt = new + out
        if tools is None:
            self._prev_prompt = 0
        text = "".join(text_parts)
        text_calls = False
        if not calls and text and tools and model_error is None:
            calls = parse_text_tool_calls(text, {t["name"] for t in tools})
            text_calls = bool(calls)
        if model_error is not None:
            usage = Usage(input_new=est_total, cache_read=0, cache_write=0, output=count_tokens(text))
            calls, text = [], ""       # keep degenerate output out of the history
            self._prev_prompt = 0
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        tcs = []
        for i, c in enumerate(calls):
            fn = c.get("function") or {}
            tid = f"toolu_ollama_{int(t0 * 1e6)}_{i}"
            args = _parse_args(fn.get("arguments"))
            content.append({"type": "tool_use", "id": tid, "name": fn.get("name", ""), "input": args})
            tcs.append(ToolCall(fn.get("name", ""), args, tid))
        if not content:
            content = [{"type": "text", "text": ""}]
        ns = 1e-9
        extra = {"prompt_eval_s": (final.get("prompt_eval_duration") or 0) * ns,
                 "eval_s": (final.get("eval_duration") or 0) * ns,
                 "load_s": (final.get("load_duration") or 0) * ns, "est_context": est_total,
                 "done_reason": final.get("done_reason"), "text_tool_calls": text_calls,
                 "model_error": model_error}
        return LLMResponse(content=content, tool_calls=tcs, text=text, usage=usage,
                           ttft_s=ttft if ttft is not None else total, total_s=total, model_id=self.model_id,
                           stop_reason=("model_error" if model_error is not None else
                                        "tool_use" if tcs else (final.get("done_reason") or "stop")),
                           extra=extra)

    def summarize(self, system, messages, *, max_tokens=2048, history_tokens=None) -> LLMResponse:
        from loopbounds.harness.llm.anthropic_client import render_transcript

        transcript = render_transcript(strip_private(messages))
        return self.complete(system, [{"role": "user", "content": transcript}], None, max_tokens=max_tokens)
