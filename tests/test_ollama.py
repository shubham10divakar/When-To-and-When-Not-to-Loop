"""Ollama client: message conversion (offline) and a live smoke test if a server is running."""

import pytest

from loopbounds.harness.llm.ollama_client import OllamaClient, to_ollama_messages, to_ollama_tools
from loopbounds.harness.tools import TOOL_SCHEMAS


def test_message_conversion():
    hist = [
        {"role": "user", "content": "start", "_ntok": 1},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "read_file",
                                           "input": {"name": "abc"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "hello"}]},
    ]
    out = to_ollama_messages("sys", hist)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"] == {"name": "read_file", "arguments": {"name": "abc"}}
    assert out[3] == {"role": "tool", "content": "hello", "tool_name": "read_file"}
    tools = to_ollama_tools(TOOL_SCHEMAS)
    assert tools[1]["function"]["parameters"]["required"] == ["name"]


def _live(model="llama3.1:8b"):
    c = OllamaClient(model, num_ctx=8192)
    return c if c.available() else None


@pytest.mark.api
def test_live_tool_call():
    c = _live()
    if c is None:
        pytest.skip("no local Ollama server with llama3.1:8b")
    r = c.complete("Use tools.", [{"role": "user", "content": "Read the file named abc123."}], TOOL_SCHEMAS,
                   max_tokens=64)
    assert r.tool_call is not None and r.tool_call.name == "read_file"
    assert r.usage.input_new > 0 and r.usage.output > 0


def test_openai_message_conversion():
    from loopbounds.harness.llm.openai_client import to_openai_messages, to_openai_tools

    hist = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "read_file",
                                           "input": {"name": "abc"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "hello"}]},
    ]
    out = to_openai_messages("sys", hist)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"name": "abc"}'
    assert out[3] == {"role": "tool", "tool_call_id": "call_1", "content": "hello"}
    assert to_openai_tools(TOOL_SCHEMAS)[2]["function"]["name"] == "answer"


def test_study_budget_ledger(tmp_path, monkeypatch):
    import json

    from loopbounds import pipeline as pl

    monkeypatch.setattr(pl, "REPO_ROOT", tmp_path)
    p = pl.Paths.make(openai="gpt-4.1-nano", budget_usd=1.0)
    (p.raw / "E1").mkdir()
    (p.raw / "E1" / "a.jsonl").write_text(json.dumps({"type": "run", "cost_usd": 0.4}) + "\n")
    assert p.spent_usd() == 0.4
    cfg = pl._cfg("E3_reliability_cal", p)
    assert cfg["client"]["kind"] == "openai" and abs(cfg["max_spend_usd"] - 0.6) < 1e-9


def test_text_tool_calls_are_recovered():
    from loopbounds.harness.llm.ollama_client import parse_text_tool_calls

    names = {"list_files", "read_file", "answer"}
    got = parse_text_tool_calls('[TOOL_CALLS]  [{"name": "read_file", "arguments": {"name": "ab12cd"}}]', names)
    assert got == [{"function": {"name": "read_file", "arguments": {"name": "ab12cd"}}}]
    got = parse_text_tool_calls('sure: {"name": "list_files", "arguments": {}} then {"name": "rm", "arguments": {}}',
                                names)
    assert [g["function"]["name"] for g in got] == ["list_files"]
    assert parse_text_tool_calls("no calls here [1, 2]", names) == []
