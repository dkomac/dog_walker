from dog_walker.storage.sqlite import SqliteStorage
from dog_walker.types import Message, ToolCall


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
