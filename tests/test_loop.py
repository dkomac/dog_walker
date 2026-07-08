from dog_walker.loop import Harness, build_system_prompt
from dog_walker.providers.fake import FakeProvider
from dog_walker.storage.sqlite import SqliteStorage
from dog_walker.tools.builtin import build_registry
from dog_walker.types import Response, ToolCall, Usage


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


def test_system_prompt_includes_preferences():
    p = build_system_prompt("/tmp", ["read_file"], preferences="- be terse\n")
    assert "be terse" in p
    assert "User preferences" in p


def test_system_prompt_without_preferences_omits_section():
    p = build_system_prompt("/tmp", ["read_file"])
    assert "User preferences" not in p


def test_verbose_traces_calls_and_results(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("hello contents")
    provider = FakeProvider([
        Response(text=None, tool_calls=[ToolCall(id="t1", name="read_file",
                                                 args={"path": str(f)})]),
        Response(text="all done", tool_calls=[]),
    ])
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5, verbose=True)
    harness.run("go")
    err = capsys.readouterr().err
    assert "read_file" in err          # the call was traced
    assert "hello contents" in err     # the result was traced


def test_non_verbose_is_silent(tmp_path, capsys):
    provider = FakeProvider([Response(text="done", tool_calls=[])])
    reg = build_registry([], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5)
    harness.run("go")
    assert capsys.readouterr().err == ""


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


def test_records_success_run(tmp_path):
    storage = _storage(tmp_path)
    provider = FakeProvider([
        Response(text="the answer", tool_calls=[], usage=Usage(input_tokens=8, output_tokens=2)),
    ])
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    harness.run("do it")
    runs = storage.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.outcome == "success"
    assert r.prompt == "do it"
    assert r.final_answer == "the answer"
    assert r.iterations == 1
    assert r.tool_calls == 0
    assert r.input_tokens == 8 and r.output_tokens == 2
    assert r.provider == "fake" and r.model == "m"
    assert r.latency_ms >= 0


def test_records_tokens_summed_and_tools_collected(tmp_path):
    storage = _storage(tmp_path)
    provider = FakeProvider([
        Response(text=None,
                 tool_calls=[ToolCall(id="c0", name="list_files", args={"path": "."})],
                 usage=Usage(input_tokens=5, output_tokens=1)),
        Response(text="done", tool_calls=[], usage=Usage(input_tokens=6, output_tokens=3)),
    ])
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    harness.run("list")
    r = storage.list_runs()[0]
    assert r.outcome == "success"
    assert r.iterations == 2
    assert r.tool_calls == 1
    assert r.tools_used == ["list_files"]
    assert r.input_tokens == 11 and r.output_tokens == 4


def test_records_max_iterations(tmp_path):
    storage = _storage(tmp_path)
    looping = [
        Response(text=None,
                 tool_calls=[ToolCall(id="c0", name="list_files", args={"path": "."})])
        for _ in range(10)
    ]
    provider = FakeProvider(looping)
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=3,
                      provider_name="fake", model="m")
    harness.run("loop forever")
    r = storage.list_runs()[0]
    assert r.outcome == "max_iterations"
    assert r.iterations == 3


def test_no_usage_reported_stores_null_tokens(tmp_path):
    storage = _storage(tmp_path)
    provider = FakeProvider([Response(text="hi", tool_calls=[])])
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    harness.run("hi")
    r = storage.list_runs()[0]
    assert r.input_tokens is None and r.output_tokens is None


def test_records_error_and_reraises(tmp_path):
    import pytest

    class BoomProvider:
        model = "m"
        def send(self, messages, tools):
            raise RuntimeError("boom")

    storage = _storage(tmp_path)
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(BoomProvider(), reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    with pytest.raises(RuntimeError, match="boom"):
        harness.run("go")
    r = storage.list_runs()[0]
    assert r.outcome == "error"
    assert "boom" in r.error
