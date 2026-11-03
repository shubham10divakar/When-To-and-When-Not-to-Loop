"""Pointer-chase tools (design doc §5.2): list_files, read_file, answer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from loopbounds.harness.injection import Injector, InjectRecord
from loopbounds.harness.taskgen import TaskInstance

TOOL_SCHEMAS = [
    {
        "name": "list_files",
        "description": "List the names of all files in the directory.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_file",
        "description": "Read one file and return its full text.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Exact file name"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "answer",
        "description": "Submit the final answer token. Ends the task.",
        "input_schema": {
            "type": "object",
            "properties": {"token": {"type": "string", "description": "The answer token"}},
            "required": ["token"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ToolResult:
    content: str
    done: bool = False
    correct: bool | None = None
    inject: InjectRecord = field(default_factory=lambda: InjectRecord(outcome="none"))
    latency_s: float = 0.0


class Toolset:
    def __init__(self, task: TaskInstance, injector: Injector | None = None):
        self.task = task
        self.injector = injector or Injector(0.0, 1.0, 0)

    def execute(self, name: str, args: dict | None) -> ToolResult:
        args = args or {}
        t = time.perf_counter()
        if name == "list_files":
            res = ToolResult("\n".join(sorted(self.task.files)))
        elif name == "read_file":
            fname = str(args.get("name", "")).strip()
            content, rec = self.injector.read(self.task, fname)
            res = ToolResult(content, inject=rec)
        elif name == "answer":
            tok = str(args.get("token", "")).strip()
            ok = tok == self.task.answer
            res = ToolResult("Answer recorded.", done=True, correct=ok)
        else:
            res = ToolResult(f"ERROR: unknown tool {name}")
        res.latency_s = time.perf_counter() - t
        return res
