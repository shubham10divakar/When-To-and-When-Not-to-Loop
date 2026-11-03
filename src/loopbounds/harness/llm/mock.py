"""Mock LLM clients for zero-cost validation (design doc §5.8).

* :class:`PerfectMock`  — honestly parses the latest observation and follows the
  valid clue; success must be 100% without injection, and logged tokens must
  exactly match the L1 naive/cached formulas.
* :class:`MarkovMock`   — behaves stochastically per the ON/DER Markov model with
  known (p, r, e); used to verify that ``analysis.fit`` recovers them.
* :class:`CoupledMock`  — like MarkovMock with p = p(T) (verifies the λ estimator).

Synthetic usage: the input of a call is ``tokens(system) + tokens(tools) +
Σ message._ntok``; output is a fixed ``a_out``. With ``cache=True`` the previous
request's input is a cache read and only the new suffix is written.
Latency is synthetic too (TTFT = α0 + α1·new + α2·cached; decode a/τ), no sleeping.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import numpy as np

from loopbounds.harness.llm.base import LLMClient, LLMResponse, ToolCall, Usage
from loopbounds.harness.tokens import count_json_tokens, count_tokens
from loopbounds.model.reliability import coupled_p

_ids = itertools.count()


def _last_tool_result_text(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg["role"] != "user":
            continue
        content = msg["content"]
        if isinstance(content, str):
            return content
        for blk in content:
            if blk.get("type") == "tool_result":
                c = blk.get("content")
                return c if isinstance(c, str) else "".join(x.get("text", "") for x in c)
            if blk.get("type") == "text":
                return blk["text"]
    return None


def parse_clue(text: str, key: str) -> tuple[str, str] | None:
    """Return ('next', name) or ('answer', token) from the valid clue lines in text."""
    if text is None:
        return None
    k = re.escape(key)
    m = re.search(rf"^\[{k}\] ANSWER: (\S+)\s*$", text, flags=re.M)
    if m:
        return "answer", m.group(1)
    m = re.search(rf"^\[{k}\] NEXT FILE: the file whose name, reversed, is (\S+)\s*$", text, flags=re.M)
    if m:
        return "next", m.group(1)[::-1]
    m1 = re.search(rf"^\[{k}\] NEXT FILE \(part 1 of 2\):.*begins with (\S+)\s*$", text, flags=re.M)
    m2 = re.search(rf"^\[{k}\] NEXT FILE \(part 2 of 2\):.*ends with (\S+)\s*$", text, flags=re.M)
    if m1 and m2:
        return "next", (m1.group(1) + m2.group(1))[::-1]
    m = re.search(rf"^\[{k}\] NEXT FILE: (\S+)\s*$", text, flags=re.M)
    if m:
        return "next", m.group(1)
    return None


class MockClient(LLMClient):
    kind = "mock"

    def __init__(self, *, a_out: int = 60, ttft_a0: float = 0.35, ttft_a1: float = 2e-5,
                 ttft_a2: float = 2e-6, tps: float = 150.0, gamma: float = 0.2,
                 ss_b0: float = 4.0, ss_b1: float = -0.15, ss_b2: float = -0.02, seed: int = 0,
                 model_id: str = "mock"):
        self.a_out = int(a_out)
        self.lat = (ttft_a0, ttft_a1, ttft_a2, tps)
        self.gamma = gamma
        self.ss = (ss_b0, ss_b1, ss_b2)
        self.seed = seed
        self.model_id = model_id
        self.rng = np.random.default_rng(seed)
        self.task = None
        self._prev_total: int | None = None
        self._n_bound = 0

    # -- hooks -------------------------------------------------------------
    def bind(self, task: Any) -> None:
        self.task = task
        self._prev_total = None
        self._reset()
        seed = int(task.config.seed) if task is not None and hasattr(task, "config") else 0
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, seed, self._n_bound, 0x5EED]))
        self._n_bound += 1

    def _reset(self) -> None:
        pass

    def policy(self, messages: list[dict], context_tokens: int) -> tuple[str, dict]:
        raise NotImplementedError

    # -- usage / latency ----------------------------------------------------
    @staticmethod
    def prompt_tokens(system: str | None, messages: list[dict], tools: list[dict] | None) -> int:
        tot = count_tokens(system or "") + (count_json_tokens(tools) if tools else 0)
        for m in messages:
            n = m.get("_ntok")
            if n is None:
                c = m["content"]
                n = count_tokens(c) if isinstance(c, str) else count_json_tokens(c)
            tot += int(n)
        return tot

    def _usage(self, total: int, cache: bool, out: int) -> Usage:
        if cache:
            if self._prev_total is None or self._prev_total > total:
                u = Usage(0, 0, total, out)
            else:
                u = Usage(0, self._prev_total, total - self._prev_total, out)
            self._prev_total = total
            return u
        return Usage(total, 0, 0, out)

    def _latency(self, u: Usage) -> tuple[float, float]:
        a0, a1, a2, tps = self.lat
        ttft = a0 + a1 * (u.input_new + u.cache_write) + a2 * u.cache_read
        return ttft, ttft + u.output / tps

    # -- API ---------------------------------------------------------------
    def complete(self, system, messages, tools, *, cache=False, max_tokens=1024) -> LLMResponse:
        total = self.prompt_tokens(system, messages, tools)
        if tools is None:  # single-shot call
            return self._single_shot(total, max_tokens)
        name, args = self.policy(messages, total)
        out = min(self.a_out, max_tokens)
        u = self._usage(total, cache, out)
        ttft, tot = self._latency(u)
        tid = f"toolu_mock_{next(_ids)}"
        block = {"type": "tool_use", "id": tid, "name": name, "input": args}
        return LLMResponse(content=[block], tool_calls=[ToolCall(name, args, tid)], text="", usage=u,
                           ttft_s=ttft, total_s=tot, model_id=self.model_id, stop_reason="tool_use")

    def summarize(self, system, messages, *, max_tokens=2048, history_tokens=None) -> LLMResponse:
        hist = int(history_tokens if history_tokens is not None else self.prompt_tokens(None, messages, None))
        total = count_tokens(system) + hist
        out = int(round(self.gamma * hist))
        last = _last_tool_result_text(messages) or ""
        keep = "\n".join(ln for ln in last.splitlines() if self.task is not None and ln.startswith(f"[{self.task.key}]"))
        body = keep + "\n" if keep else ""
        pad = max(0, out * 4 - len(body))
        text = body + ("." * pad)
        u = Usage(total, 0, 0, out)
        ttft, tot = self._latency(u)
        return LLMResponse(content=[{"type": "text", "text": text}], tool_calls=[], text=text, usage=u,
                           ttft_s=ttft, total_s=tot, model_id=self.model_id, stop_reason="end_turn")

    def _single_shot(self, total: int, max_tokens: int) -> LLMResponse:
        task = self.task
        b0, b1, b2 = self.ss
        q = 1.0 / (1.0 + np.exp(-(b0 + b1 * task.depth + b2 * total / 1000.0)))
        tok = task.answer if self.rng.random() < q else (task.decoy_answers[0] if task.decoy_answers else "WRONG000")
        text = f"ANSWER: {tok}"
        out = min(count_tokens(text), max_tokens)
        u = Usage(total, 0, 0, out)
        ttft, tot = self._latency(u)
        return LLMResponse(content=[{"type": "text", "text": text}], tool_calls=[], text=text, usage=u,
                           ttft_s=ttft, total_s=tot, model_id=self.model_id, stop_reason="end_turn",
                           extra={"q": q})


class PerfectMock(MockClient):
    kind = "perfect_mock"

    def _reset(self):
        self._last_read: str | None = None

    def policy(self, messages, context_tokens):
        task = self.task
        text = _last_tool_result_text(messages)
        if self._last_read is None:
            self._last_read = task.start
            return "read_file", {"name": task.start}
        clue = parse_clue(text, task.key)
        if clue is None:          # error / unparseable -> re-read
            return "read_file", {"name": self._last_read}
        kind, val = clue
        if kind == "answer":
            return "answer", {"token": val}
        self._last_read = val
        return "read_file", {"name": val}


class MarkovMock(MockClient):
    """Ground-truth-aware stochastic policy implementing the ON/DER Markov model."""
    kind = "markov_mock"

    def __init__(self, p: float = 0.95, r: float = 0.3, e: float = 0.05, **kw):
        super().__init__(**kw)
        self.p, self.r, self.e = float(p), float(r), float(e)

    def _reset(self):
        self.progress = 0
        self.state = "ON"
        self._req: str | None = None

    def _p(self, context_tokens: int) -> float:
        return self.p

    def summarize(self, system, messages, **kw):
        self._observe(messages)  # the history is about to be replaced; record what was read
        return super().summarize(system, messages, **kw)

    def _observe(self, messages):
        if self._req is None:
            return
        task = self.task
        text = _last_tool_result_text(messages)
        if self.progress < task.depth and self._req == task.chain[self.progress]:
            if text == task.files[self._req]:
                self.progress += 1
                self.state = "ON"
            elif text == task.silent_files.get(self._req):
                self.state = "DER"
        self._req = None

    def _wrong_file(self) -> str:
        task = self.task
        pool = [x for dc in task.decoys for x in dc]
        return str(self.rng.choice(pool)) if pool else "zzzzzz"

    def policy(self, messages, context_tokens):
        self._observe(messages)
        task = self.task
        if self.progress >= task.depth:
            return "answer", {"token": task.answer}
        u = self.rng.random()
        if self.state == "ON":
            if u < self._p(context_tokens):
                self._req = task.chain[self.progress]
                return "read_file", {"name": self._req}
            self.state = "DER"
            return "read_file", {"name": self._wrong_file()}
        # DER
        if u < self.r:
            self._req = task.chain[self.progress]
            return "read_file", {"name": self._req}
        if u < self.r + self.e:
            wrong = task.decoy_answers[0] if task.decoy_answers else "WRONG000"
            return "answer", {"token": wrong}
        return "read_file", {"name": self._wrong_file()}


class CoupledMock(MarkovMock):
    kind = "coupled_mock"

    def __init__(self, p0: float = 0.97, lam: float = 1e-6, form: str = "exp", r: float = 0.3,
                 e: float = 0.05, **kw):
        super().__init__(p=p0, r=r, e=e, **kw)
        self.p0, self.lam, self.form = float(p0), float(lam), form

    def _p(self, context_tokens: int) -> float:
        return float(coupled_p(context_tokens, self.p0, self.lam, self.form))
