"""Step classification against the ground-truth chain (design doc §5.5).

Labels:
  ADVANCE         read chain[progress] and received the true content (progress += 1)
  CORRUPTED       read chain[progress] but a silent failure redirected the clue (→ DER)
  ERROR           read chain[progress] but retries were exhausted (no state change)
  REDUNDANT       re-read an already-visited chain file
  WRONG           read a non-chain file or skipped ahead (→ DER)
  LIST            list_files (neutral)
  NOOP            no tool call / unknown tool (neutral)
  ANSWER_CORRECT / ANSWER_WRONG

Markov state: DER after any WRONG or CORRUPTED step until the next ADVANCE; ON
otherwise. Each step is recorded with the state it was taken *from* and a
``phase`` ("chase" while progress < n, "answer" once the answer is in hand).

Estimators (fixed in PREREG.md), over chase-phase steps:
  p̂ = #ADVANCE-from-ON / #steps-from-ON
  r̂ = #ADVANCE-from-DER / #steps-from-DER
  ê = #ANSWER_WRONG-from-DER / #steps-from-DER
"""

from __future__ import annotations

from dataclasses import dataclass

from loopbounds.harness.taskgen import TaskInstance

LABELS = ("ADVANCE", "CORRUPTED", "ERROR", "REDUNDANT", "WRONG", "LIST", "NOOP",
          "ANSWER_CORRECT", "ANSWER_WRONG")


@dataclass
class StepLabel:
    label: str
    state_from: str       # ON | DER
    state_to: str
    phase: str            # chase | answer
    progress: int         # progress after this step


class Classifier:
    def __init__(self, task: TaskInstance):
        self.task = task
        self.progress = 0
        self.state = "ON"

    @property
    def n(self) -> int:
        return self.task.depth

    def step(self, tool: str | None, args: dict | None, inject_outcome: str = "clean") -> StepLabel:
        args = args or {}
        s0 = self.state
        phase = "answer" if self.progress >= self.n else "chase"
        if tool == "list_files":
            label = "LIST"
        elif tool == "read_file":
            name = str(args.get("name", "")).strip()
            idx = self.task.index_of(name)
            if idx is not None and idx == self.progress:
                if inject_outcome == "silent":
                    label = "CORRUPTED"
                    self.state = "DER"
                elif inject_outcome == "exhausted":
                    label = "ERROR"
                else:
                    label = "ADVANCE"
                    self.progress += 1
                    self.state = "ON"
            elif idx is not None and idx < self.progress:
                label = "REDUNDANT"
            else:
                label = "WRONG"
                self.state = "DER"
        elif tool == "answer":
            label = "ANSWER_CORRECT" if str(args.get("token", "")).strip() == self.task.answer else "ANSWER_WRONG"
        else:
            label = "NOOP"
        return StepLabel(label, s0, self.state, phase, self.progress)
