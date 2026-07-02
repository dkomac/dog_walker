from types import SimpleNamespace
from middlemanager.providers.anthropic import (
    to_anthropic_messages, parse_anthropic_response, extract_system)
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


def test_system_message_excluded_from_messages():
    # Anthropic takes system as a separate param, so it must NOT appear in messages.
    msgs = [Message(role="system", text="sys"), Message(role="user", text="hi")]
    assert to_anthropic_messages(msgs) == [{"role": "user", "content": "hi"}]


def test_extract_system_returns_system_text():
    msgs = [Message(role="system", text="be good"), Message(role="user", text="hi")]
    assert extract_system(msgs) == "be good"


def test_extract_system_none_when_absent():
    assert extract_system([Message(role="user", text="hi")]) is None


def test_parse_response_with_text_and_tool_use():
    blocks = [
        SimpleNamespace(type="text", text="let me check"),
        SimpleNamespace(type="tool_use", id="t9", name="run_bash", input={"command": "ls"}),
    ]
    resp = parse_anthropic_response(blocks)
    assert resp.text == "let me check"
    assert resp.tool_calls == [ToolCall(id="t9", name="run_bash", args={"command": "ls"})]
