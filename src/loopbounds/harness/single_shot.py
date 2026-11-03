"""Single-shot baseline runner (design doc §5.6).

One call with every file (name + content) plus the task. If the estimated
context exceeds the window, the run is recorded as ``infeasible`` and no call
is made.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from loopbounds.harness import taskgen
from loopbounds.harness.llm.base import ContextOverflow, LLMClient, Usage
from loopbounds.harness.tokens import count_tokens

SYSTEM = "You are a careful assistant. Answer with exactly one line of the form 'ANSWER: <token>'."


@dataclass
class SingleShotResult:
    feasible: bool
    success: bool
    answer: str | None
    usage: Usage
    ttft_s: float
    total_s: float
    context_estimate: int
    model_id: str
    rate_limit_wait_s: float = 0.0


def run_single_shot(task: taskgen.TaskInstance, llm: LLMClient, *, window: int, max_tokens: int = 64,
                    seed: int = 0) -> SingleShotResult:
    llm.bind(task)
    prompt = taskgen.single_shot_prompt(task, np.random.default_rng(seed))
    est = count_tokens(SYSTEM) + count_tokens(prompt)
    if est > window:
        return SingleShotResult(False, False, None, Usage(), 0.0, 0.0, est, getattr(llm, "model_id", ""))
    try:
        resp = llm.complete(SYSTEM, [{"role": "user", "content": prompt, "_ntok": count_tokens(prompt)}], None,
                            cache=False, max_tokens=max_tokens)
    except ContextOverflow:
        return SingleShotResult(False, False, None, Usage(), 0.0, 0.0, est, getattr(llm, "model_id", ""))
    m = re.search(r"ANSWER:\s*(\S+)", resp.text or "")
    ans = m.group(1).strip() if m else None
    return SingleShotResult(True, ans == task.answer, ans, resp.usage, resp.ttft_s, resp.total_s, est,
                            resp.model_id, resp.rate_limit_wait_s)
