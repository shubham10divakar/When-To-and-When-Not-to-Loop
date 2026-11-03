"""Instrumented ReAct-style tool-use loop (design doc §5.3).

The loop is generic over an *environment* (pointer-chase here, coding tasks in
``loopbounds.tasks_coding``) that provides tool schemas, prompts, tool
execution and per-step labelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from loopbounds.harness import taskgen
from loopbounds.harness.arms import Arm
from loopbounds.harness.classify import Classifier
from loopbounds.harness.injection import Injector
from loopbounds.harness.llm.base import ContextOverflow, LLMClient, Usage
from loopbounds.harness.logging import InjectInfo, StepRecord, now_iso
from loopbounds.harness.tokens import count_tokens
from loopbounds.harness.tools import TOOL_SCHEMAS, Toolset, ToolResult

NUDGE = "Please continue by calling exactly one of the available tools."
ONE_CALL = "ERROR: only one tool call is executed per step; this call was skipped."


class Environment(Protocol):
    task: Any
    tools: list[dict]

    def system_prompt(self) -> str: ...
    def task_prompt(self) -> str: ...
    def execute(self, name: str, args: dict) -> ToolResult: ...
    def label(self, name: str | None, args: dict | None, result: ToolResult | None) -> dict: ...


class PointerChaseEnv:
    def __init__(self, task: taskgen.TaskInstance, injector: Injector | None = None):
        self.task = task
        self.tools = TOOL_SCHEMAS
        self.toolset = Toolset(task, injector)
        self.clf = Classifier(task)

    def system_prompt(self) -> str:
        return taskgen.system_prompt(self.task)

    def task_prompt(self) -> str:
        return taskgen.task_prompt(self.task)

    def execute(self, name, args):
        return self.toolset.execute(name, args)

    def label(self, name, args, result):
        outcome = result.inject.outcome if result is not None else "none"
        sl = self.clf.step(name, args, outcome)
        return {"label": sl.label, "markov_state": sl.state_from, "state_to": sl.state_to,
                "phase": sl.phase, "progress": sl.progress}


@dataclass
class AgentResult:
    steps: list[StepRecord]
    success: bool
    termination: str
    usage: Usage
    summary_usage: Usage
    wall_s: float
    rate_limit_wait_s: float
    t0_tokens: int | None
    model_id: str
    extra: dict = field(default_factory=dict)


def _tool_arg(args: dict | None) -> str | None:
    if not args:
        return None
    for k in ("name", "token", "path"):
        if k in args:
            return str(args[k])
    return str(args)[:200]


def run_agent(env: Environment, llm: LLMClient, arm: Arm, *, max_steps: int, run_id: str,
              window_cap: int | None = None, max_tokens: int = 1024, spend_hook=None) -> AgentResult:
    """Run one episode. ``spend_hook(usage)`` may raise to abort (budget guard)."""
    llm.bind(env.task)
    system = env.system_prompt()
    tools = env.tools
    task_text = env.task_prompt()
    task_ntok = count_tokens(task_text)
    messages: list[dict] = [{"role": "user", "content": task_text, "_ntok": task_ntok}]
    steps: list[StepRecord] = []
    usage, s_usage = Usage(), Usage()
    wall = wait = 0.0
    termination, success = "budget", False
    t0_tokens = None
    model_id = getattr(llm, "model_id", "")
    pending_summary = None
    est_next: int | None = None
    fixed_overhead: int | None = None   # system + tools tokens (estimated from call 0)

    for step in range(max_steps):
        if window_cap is not None and est_next is not None and est_next > window_cap:
            termination = "context_overflow"
            break
        try:
            resp = llm.complete(system, messages, tools, cache=arm.cache, max_tokens=max_tokens)
        except ContextOverflow:
            termination = "context_overflow"
            break
        model_id = resp.model_id or model_id
        usage = usage + resp.usage
        if spend_hook is not None:
            spend_hook(resp.usage)
        if t0_tokens is None:
            t0_tokens = resp.usage.context
            fixed_overhead = max(0, resp.usage.context - task_ntok)
        call = resp.tool_call
        result = env.execute(call.name, call.input) if call is not None else None
        lab = env.label(call.name if call else None, call.input if call else None, result)
        inj = result.inject if result is not None else None
        rec = StepRecord(
            run_id=run_id, step=step, arm=arm.name, context_tokens=resp.usage.context,
            input_tokens_new=resp.usage.input_new, cache_read_tokens=resp.usage.cache_read,
            cache_write_tokens=resp.usage.cache_write, output_tokens=resp.usage.output,
            ttft_s=resp.ttft_s, llm_total_s=resp.total_s,
            tool_name=call.name if call else None, tool_arg=_tool_arg(call.input) if call else None,
            tool_latency_s=result.latency_s if result else 0.0,
            inject=InjectInfo(**inj.as_dict()) if inj is not None else InjectInfo(),
            rate_limit_wait_s=resp.rate_limit_wait_s, stop_reason=resp.stop_reason, model_id=resp.model_id,
            ts=now_iso(), **lab,
        )
        if pending_summary is not None:
            rec.summary_input_tokens = pending_summary.input_tokens
            rec.summary_output_tokens = pending_summary.output_tokens
            rec.summary_latency_s = pending_summary.latency_s
            pending_summary = None
        steps.append(rec)
        wall += resp.total_s + rec.tool_latency_s + rec.summary_latency_s
        wait += resp.rate_limit_wait_s

        messages.append({"role": "assistant", "content": resp.content, "_ntok": resp.usage.output})
        if result is not None and result.done:
            success = bool(result.correct)
            termination = "answer_correct" if success else "answer_wrong"
            break
        if call is not None:
            blocks = [{"type": "tool_result", "tool_use_id": call.id, "content": result.content}]
            for extra in resp.tool_calls[1:]:
                blocks.append({"type": "tool_result", "tool_use_id": extra.id, "content": ONE_CALL,
                               "is_error": True})
            user_ntok = count_tokens(result.content) + sum(count_tokens(ONE_CALL) for _ in resp.tool_calls[1:])
            messages.append({"role": "user", "content": blocks, "_ntok": user_ntok})
        else:
            user_ntok = count_tokens(NUDGE)
            messages.append({"role": "user", "content": NUDGE, "_ntok": user_ntok})
        est_next = resp.usage.context + resp.usage.output + user_ntok

        if arm.should_summarize(step, max_steps):
            messages, pending_summary = arm.summarize(llm, messages, task_text, task_ntok)
            s_usage = s_usage + Usage(pending_summary.input_tokens, 0, 0, pending_summary.output_tokens)
            if spend_hook is not None:
                spend_hook(Usage(pending_summary.input_tokens, 0, 0, pending_summary.output_tokens))
            wait += pending_summary.rate_limit_wait_s
            est_next = (fixed_overhead or 0) + sum(int(m.get("_ntok", 0)) for m in messages)

    return AgentResult(steps, success, termination, usage, s_usage, wall, wait, t0_tokens, model_id)
