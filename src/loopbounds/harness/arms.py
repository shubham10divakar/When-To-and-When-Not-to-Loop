"""Context strategies (design doc §5.3): naive, cached, summarized."""

from __future__ import annotations

from dataclasses import dataclass

from loopbounds.harness.llm.base import LLMClient, LLMResponse
from loopbounds.harness.tokens import count_tokens

SUMMARY_HEADER = "[Progress summary of the work so far]\n"

SUMMARY_SYSTEM = (
    "You maintain a compact progress log for an agent that navigates a chain of text files by "
    "following tagged clue lines. Summarize the transcript you are given. You MUST preserve, verbatim: "
    "(1) the name of the last file that was read successfully, (2) the exact text of the most recent "
    "valid clue line(s) that tell the agent which file to read next (or the ANSWER line if found), and "
    "(3) the list of file names already visited. Omit filler text. Be concise."
)


@dataclass
class SummaryInfo:
    input_tokens: int
    output_tokens: int
    latency_s: float
    rate_limit_wait_s: float = 0.0
    model_id: str = ""


class Arm:
    name = "base"
    cache = False

    def should_summarize(self, step: int, max_steps: int) -> bool:
        return False

    def summarize(self, llm: LLMClient, messages: list[dict], task_text: str, task_ntok: int):
        raise NotImplementedError


class NaiveArm(Arm):
    name = "naive"
    cache = False


class CachedArm(Arm):
    name = "cached"
    cache = True


class SummarizedArm(Arm):
    """Every k calls, replace the history since T0 with an LLM-written summary."""
    name = "summarized"
    cache = False

    def __init__(self, k: int = 5, system: str = SUMMARY_SYSTEM, max_tokens: int = 2048):
        self.k = int(k)
        self.system = system
        self.max_tokens = max_tokens

    def should_summarize(self, step: int, max_steps: int) -> bool:
        # the summary precedes call step+1 when (step+1) % k == 0 and that call exists
        nxt = step + 1
        return self.k > 0 and nxt % self.k == 0 and nxt < max_steps

    def summarize(self, llm: LLMClient, messages: list[dict], task_text: str, task_ntok: int):
        hist_tokens = sum(int(m.get("_ntok", 0)) for m in messages) - task_ntok
        resp: LLMResponse = llm.summarize(self.system, messages, max_tokens=self.max_tokens,
                                          history_tokens=hist_tokens)
        summary = resp.text.strip()
        content = f"{task_text}\n\n{SUMMARY_HEADER}{summary}"
        ntok = task_ntok + count_tokens(SUMMARY_HEADER) + resp.usage.output
        new_messages = [{"role": "user", "content": content, "_ntok": ntok}]
        info = SummaryInfo(resp.usage.context, resp.usage.output, resp.total_s, resp.rate_limit_wait_s,
                           resp.model_id)
        return new_messages, info


def make_arm(name: str, summ_every: int = 5) -> Arm:
    if name == "naive":
        return NaiveArm()
    if name == "cached":
        return CachedArm()
    if name == "summarized":
        return SummarizedArm(summ_every)
    raise ValueError(f"unknown arm {name!r}")
