from dog_walker.providers.fake import FakeProvider
from dog_walker.types import Response, ToolCall, Message


def test_fake_returns_scripted_responses_in_order():
    p = FakeProvider([
        Response(text=None, tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})]),
        Response(text="done", tool_calls=[]),
    ])
    r1 = p.send([Message(role="user", text="hi")], [])
    r2 = p.send([Message(role="user", text="hi")], [])
    assert r1.tool_calls[0].name == "read_file"
    assert r2.text == "done"


def test_fake_records_calls():
    p = FakeProvider([Response(text="ok", tool_calls=[])])
    p.send([Message(role="user", text="hi")], [])
    assert p.calls[0][0].text == "hi"
