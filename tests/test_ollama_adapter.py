from types import SimpleNamespace
from middlemanager.providers.ollama import to_ollama_messages, to_ollama_tools, parse_ollama_response
from middlemanager.types import Message, ToolCall, ToolResult, ToolSpec


def test_user_message_translation():
    out = to_ollama_messages([Message(role="user", text="hi")])
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_with_tool_call_translation():
    msg = Message(role="assistant", text="looking",
                  tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})])
    out = to_ollama_messages([msg])[0]
    assert out["role"] == "assistant"
    assert out["content"] == "looking"
    assert out["tool_calls"] == [
        {"function": {"name": "read_file", "arguments": {"path": "a"}}}]


def test_tool_result_translation():
    # Ollama tool results are plain role="tool" messages, no tool_use_id wrapper.
    msg = Message(role="tool", tool_results=[ToolResult(tool_call_id="t1", content="hello")])
    out = to_ollama_messages([msg])
    assert out == [{"role": "tool", "content": "hello"}]


def test_tools_translation():
    spec = ToolSpec(name="read_file", description="reads a file",
                    parameters={"type": "object", "properties": {}})
    out = to_ollama_tools([spec])
    assert out == [{"type": "function", "function": {
        "name": "read_file", "description": "reads a file",
        "parameters": {"type": "object", "properties": {}}}}]


def test_parse_response_with_text_and_tool_use():
    message = SimpleNamespace(
        content="let me check",
        tool_calls=[SimpleNamespace(
            function=SimpleNamespace(name="run_bash", arguments={"command": "ls"}))],
    )
    resp = parse_ollama_response(message)
    assert resp.text == "let me check"
    # Ollama gives no call id, so the adapter synthesizes one.
    assert resp.tool_calls == [ToolCall(id="call_0", name="run_bash", args={"command": "ls"})]


def test_parse_response_no_tool_calls():
    message = SimpleNamespace(content="all done", tool_calls=None)
    resp = parse_ollama_response(message)
    assert resp.text == "all done"
    assert resp.tool_calls == []
