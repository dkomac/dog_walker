from dog_walker.storage.sqlite import SqliteStorage
from dog_walker.types import Message, ToolCall, RunRecord


def _rec(**over):
    base = dict(
        session_id="1", provider="fake", model="m", prompt="p",
        outcome="success", iterations=1, tool_calls=0, tools_used=[],
        latency_ms=5,
    )
    base.update(over)
    return RunRecord(**base)


def test_save_and_load_roundtrip(tmp_path):
    db = str(tmp_path / "t.db")
    store = SqliteStorage(db)
    sid = store.create_session()

    store.save_message(sid, Message(role="user", text="hi"))
    store.save_message(
        sid,
        Message(role="assistant", text="ok",
                tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})]),
    )

    loaded = store.load_messages(sid)
    assert len(loaded) == 2
    assert loaded[0] == Message(role="user", text="hi")
    assert loaded[1].tool_calls[0].name == "read_file"


def test_sessions_are_isolated(tmp_path):
    store = SqliteStorage(str(tmp_path / "t.db"))
    s1 = store.create_session()
    s2 = store.create_session()
    store.save_message(s1, Message(role="user", text="in s1"))
    assert store.load_messages(s2) == []
    assert s1 != s2


def test_record_and_list_roundtrip(tmp_path):
    s = SqliteStorage(str(tmp_path / "t.db"))
    rid = s.record_run(_rec(
        prompt="hello", tools_used=["list_files", "run_bash"],
        input_tokens=10, output_tokens=4, tool_calls=2,
    ))
    assert rid is not None
    runs = s.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.prompt == "hello"
    assert r.tools_used == ["list_files", "run_bash"]
    assert r.input_tokens == 10 and r.output_tokens == 4
    assert r.tool_calls == 2
    assert r.id is not None
    assert r.created_at is not None
    assert r.cost_usd is None


def test_list_runs_most_recent_first_and_limit(tmp_path):
    s = SqliteStorage(str(tmp_path / "t.db"))
    for i in range(3):
        s.record_run(_rec(prompt=f"p{i}"))
    runs = s.list_runs(limit=2)
    assert len(runs) == 2
    assert runs[0].prompt == "p2"   # newest first
    assert runs[1].prompt == "p1"


def test_null_tokens_roundtrip(tmp_path):
    s = SqliteStorage(str(tmp_path / "t.db"))
    s.record_run(_rec(input_tokens=None, output_tokens=None))
    r = s.list_runs()[0]
    assert r.input_tokens is None and r.output_tokens is None
