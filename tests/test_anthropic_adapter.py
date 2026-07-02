from types import SimpleNamespace
from middlemanager.providers.anthropic import to_anthropic_messages, parse_anthropic_response
from middlemanager.types import Message, ToolCall, ToolResult


def test_user_message_translation():
    out = to_anthropic_messages([Message(role="user", text="hi")])
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_with_tool_call_translation():
    msg = Message(role="assistant", text="looking",
                  tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})])
    out = to_anthropic_messages([msg])[0]
    assert out["role"] == "assistant"
    assert {"type": "text", "text": "looking"} in out["content"]
    assert {"type": "tool_use", "id": "t1", "name": "read_file",
            "input": {"path": "a"}} in out["content"]


def test_tool_result_translation():
    msg = Message(role="tool", tool_results=[ToolResult(tool_call_id="t1", content="hello")])
    out = to_anthropic_messages([msg])[0]
    assert out == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "hello"}]}


def test_parse_response_with_text_and_tool_use():
    blocks = [
        SimpleNamespace(type="text", text="let me check"),
        SimpleNamespace(type="tool_use", id="t9", name="run_bash", input={"command": "ls"}),
    ]
    resp = parse_anthropic_response(blocks)
    assert resp.text == "let me check"
    assert resp.tool_calls == [ToolCall(id="t9", name="run_bash", args={"command": "ls"})]
