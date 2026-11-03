"""Failure injection and tool-layer retry policy (design doc §5.4).

Applied only to ``read_file`` calls on true-chain files. Per attempt, an
independent draw from an RNG stream separate from task generation:
  * w.p. (1-f): the true content            -> outcome "clean"
  * w.p. f·d  : detectable failure, retried by the tool layer up to R times;
                if exhausted, the error string is returned -> "exhausted"
  * w.p. f(1-d): silent failure — content with the valid clue redirected into a
                decoy chain                                  -> "silent"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from loopbounds.harness.taskgen import TaskInstance

TRANSIENT_ERROR = "ERROR: transient read failure"


@dataclass
class InjectRecord:
    attempts: int = 0
    types: list[str] = field(default_factory=list)   # per attempt: clean | detectable | silent
    outcome: str = "clean"                            # clean | silent | exhausted | none (not injected)

    def as_dict(self) -> dict:
        return {"attempts": self.attempts, "types": list(self.types), "outcome": self.outcome}


class Injector:
    def __init__(self, f: float = 0.0, d: float = 1.0, R: int = 0, seed: int | None = 0):
        if not (0 <= f <= 1 and 0 <= d <= 1 and R >= 0):
            raise ValueError("invalid injection parameters")
        self.f, self.d, self.R = float(f), float(d), int(R)
        self.rng = np.random.default_rng(np.random.SeedSequence([0xFA11, 0 if seed is None else seed]))

    def draw(self) -> InjectRecord:
        rec = InjectRecord()
        for _ in range(self.R + 1):
            rec.attempts += 1
            u = self.rng.random()
            if u >= self.f:
                rec.types.append("clean")
                rec.outcome = "clean"
                return rec
            if self.rng.random() < self.d:
                rec.types.append("detectable")
                continue
            rec.types.append("silent")
            rec.outcome = "silent"
            return rec
        rec.outcome = "exhausted"
        return rec

    def read(self, task: TaskInstance, name: str) -> tuple[str, InjectRecord]:
        if name not in task.files:
            return f"ERROR: no such file: {name}", InjectRecord(attempts=1, types=["missing"], outcome="none")
        if task.index_of(name) is None:
            return task.files[name], InjectRecord(attempts=1, types=["clean"], outcome="none")
        rec = self.draw()
        if rec.outcome == "clean":
            return task.files[name], rec
        if rec.outcome == "silent":
            return task.silent_files[name], rec
        return TRANSIENT_ERROR, rec
