from middlemanager.types import Message, ToolCall, ToolResult


def test_message_roundtrips_through_dict():
    msg = Message(
        role="assistant",
        text="let me look",
        tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a.txt"})],
    )
    restored = Message.from_dict(msg.to_dict())
    assert restored == msg


def test_tool_message_roundtrips():
    msg = Message(role="tool", tool_results=[ToolResult(tool_call_id="t1", content="hello")])
    assert Message.from_dict(msg.to_dict()) == msg
