from dog_walker.types import Message, ToolCall, ToolResult, Usage, Response, RunRecord


def test_response_usage_defaults_none():
    r = Response(text="hi", tool_calls=[])
    assert r.usage is None


def test_response_carries_usage():
    r = Response(text="hi", tool_calls=[], usage=Usage(input_tokens=10, output_tokens=3))
    assert r.usage.input_tokens == 10
    assert r.usage.output_tokens == 3


def test_usage_defaults_none():
    u = Usage()
    assert u.input_tokens is None and u.output_tokens is None


def test_runrecord_minimal_and_defaults():
    rec = RunRecord(
        session_id="1", provider="fake", model="m", prompt="p",
        outcome="success", iterations=1, tool_calls=0, tools_used=[],
        latency_ms=5,
    )
    assert rec.final_answer is None
    assert rec.error is None
    assert rec.input_tokens is None and rec.output_tokens is None
    assert rec.cost_usd is None
    assert rec.id is None and rec.created_at is None
    assert rec.tools_used == []


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
