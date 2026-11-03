"""SimPy agent process (design doc §4.1).

Runs the same ON/DER state machine as L1 (§3.3.2) with the same parameters, but
*samples* every event: the policy's choice, each tool attempt's failure type,
retries, LLM and tool service times. Context is tracked with an independent
incremental implementation of the three arms (not by calling L1's formulas).
"""

from __future__ import annotations

import numpy as np
import simpy

from loopbounds.model.reliability import coupled_p
from loopbounds.params import Params
from loopbounds.sim.environment import LLMServer, ToolServer


class AgentProcess:
    def __init__(self, env: simpy.Environment, params: Params, arm: str, llm: LLMServer, tools: ToolServer,
                 rng: np.random.Generator):
        if arm not in ("naive", "cached", "summarized"):
            raise ValueError(arm)
        self.env, self.P, self.arm, self.llm, self.tools, self.rng = env, params, arm, llm, tools, rng

    def _p(self, ctx: float) -> float:
        P = self.P
        if P.coupling in (None, "none"):
            return P.p
        return float(coupled_p(ctx, P.p0, P.lam, P.coupling))

    def run(self, run_id: str, out: list):
        P, rng = self.P, self.rng
        n, M = int(P.depth), int(P.M)
        st = {"rate_limit_wait_s": 0.0, "queue_wait_s": 0.0, "llm_s": 0.0, "tool_s": 0.0, "tool_attempts": 0}
        tok = {"input_new": 0.0, "cache_read": 0.0, "cache_write": 0.0, "output": 0.0,
               "summary_input": 0.0, "summary_output": 0.0}
        t_begin = self.env.now
        k, state = 0, "ON"
        hist = 0.0                 # tokens in context beyond T0
        prev_ctx = None            # cached arm: previous call's full context
        steps = 0
        termination = "budget"
        for t in range(M):
            ctx = P.t0 + hist
            if ctx > P.window:
                termination = "context_overflow"
                break
            # --- LLM call ------------------------------------------------
            if self.arm == "cached":
                if prev_ctx is None:
                    t_new, t_cached, write = 0.0, 0.0, ctx
                else:
                    t_new, t_cached, write = 0.0, prev_ctx, ctx - prev_ctx
                tok["cache_read"] += t_cached
                tok["cache_write"] += write
                prev_ctx = ctx
                yield from self.llm.call(write, t_cached, P.a_out, st)
            else:
                tok["input_new"] += ctx
                yield from self.llm.call(ctx, 0.0, P.a_out, st)
            tok["output"] += P.a_out
            steps += 1
            # --- policy ------------------------------------------------------
            if k >= n:
                termination = "answer_correct"
                break
            u = rng.random()
            correct_read = False
            if state == "ON":
                if u < self._p(ctx):
                    correct_read = True
                else:
                    state = "DER"
            else:
                if u < P.r:
                    correct_read = True
                elif u < P.r + P.e:
                    termination = "answer_wrong"
                    break
            outcome = yield from self.tools.read(correct_read, st)
            if correct_read:
                if outcome == "clean":
                    k += 1
                    state = "ON"
                elif outcome == "silent":
                    state = "DER"
            hist += P.delta
            # --- summarization -----------------------------------------------
            ks = int(P.summ_every)
            if self.arm == "summarized" and ks > 0 and (t + 1) % ks == 0 and t + 1 < M:
                s_in = P.t0_summ + hist
                s_out = P.gamma * hist
                tok["summary_input"] += s_in
                tok["summary_output"] += s_out
                tok["input_new"] += s_in
                tok["output"] += s_out
                yield from self.llm.call(s_in, 0.0, s_out, st)
                hist = s_out
        cost = P.price_in * (tok["input_new"] + P.kappa * tok["cache_read"] + (1 + P.omega) * tok["cache_write"]) \
            + P.price_out * tok["output"] + P.tool_cost_usd * st["tool_attempts"]
        out.append({
            "run_id": run_id, "depth": n, "arm": self.arm, "success": termination == "answer_correct",
            "termination": termination, "steps": steps,
            **{f"tokens.{key}": v for key, v in tok.items()},
            "cost_usd": cost, "wall_s": st["llm_s"] + st["tool_s"], "e2e_s": self.env.now - t_begin,
            "queue_wait_s": st["queue_wait_s"], "rate_limit_wait_s": st["rate_limit_wait_s"],
            "tool_attempts": st["tool_attempts"],
        })
