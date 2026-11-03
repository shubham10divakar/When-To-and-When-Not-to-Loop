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
