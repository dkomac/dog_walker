from middlemanager.loop import Harness
from middlemanager.providers.fake import FakeProvider
from middlemanager.storage.sqlite import SqliteStorage
from middlemanager.tools.builtin import build_registry
from middlemanager.types import Response, ToolCall


def _storage(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def test_returns_text_when_no_tool_calls(tmp_path):
    provider = FakeProvider([Response(text="hello there", tool_calls=[])])
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5)
    assert harness.run("hi") == "hello there"


def test_runs_tool_then_finishes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("file contents")
    provider = FakeProvider([
        Response(text=None, tool_calls=[ToolCall(id="t1", name="read_file",
                                                 args={"path": str(f)})]),
        Response(text="the file says: file contents", tool_calls=[]),
    ])
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5)
    result = harness.run("read the file")
    assert result == "the file says: file contents"
    # Second call to the provider must include the tool result in the messages.
    second_call_messages = provider.calls[1]
    assert any(m.role == "tool" and m.tool_results[0].content == "file contents"
               for m in second_call_messages)


def test_prepends_system_prompt(tmp_path):
    provider = FakeProvider([Response(text="hi", tool_calls=[])])
    reg = build_registry([], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5,
                      system_prompt="BE A GOOD AGENT")
    harness.run("hello")
    first_message = provider.calls[0][0]
    assert first_message.role == "system"
    assert first_message.text == "BE A GOOD AGENT"


def test_stops_at_max_iterations(tmp_path):
    # Provider always asks for another tool call -> never terminates on its own.
    looping = [Response(text=None, tool_calls=[ToolCall(id=f"t{i}", name="read_file",
                                                        args={"path": "/nope"})])
               for i in range(10)]
    provider = FakeProvider(looping)
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=3)
    result = harness.run("go")
    assert "max iterations" in result
