import pytest
from dog_walker.config import Config
from dog_walker.main import build_storage, build_provider
from dog_walker.storage.sqlite import SqliteStorage


def _cfg(**over):
    base = dict(provider_name="anthropic", model="m", max_tokens=10,
                max_iterations=5, confirm_bash=False,
                enabled_tools=["read_file"], storage_backend="sqlite",
                storage_path=":memory:")
    base.update(over)
    return Config(**base)


def test_build_storage_sqlite():
    assert isinstance(build_storage(_cfg()), SqliteStorage)


def test_build_storage_unknown_raises():
    with pytest.raises(ValueError):
        build_storage(_cfg(storage_backend="mongo"))


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError):
        build_provider(_cfg(provider_name="openai"))


def test_build_provider_ollama():
    from dog_walker.providers.ollama import OllamaProvider
    provider = build_provider(_cfg(provider_name="ollama", model="llama3.1:8b"))
    assert isinstance(provider, OllamaProvider)


def test_build_harness_passes_provider_and_model(tmp_path, monkeypatch):
    from dog_walker.config import Config
    from dog_walker import main as main_mod

    cfg = Config(
        provider_name="ollama", model="qwen2.5:7b", max_tokens=256,
        max_iterations=5, confirm_bash=False, enabled_tools=["list_files"],
        storage_backend="sqlite", storage_path=str(tmp_path / "d.db"),
    )
    # Avoid constructing a real Ollama client.
    monkeypatch.setattr(main_mod, "build_provider", lambda c: object())
    harness = main_mod.build_harness(cfg)
    assert harness.provider_name == "ollama"
    assert harness.model == "qwen2.5:7b"


def test_render_runs_table_empty():
    from dog_walker.main import render_runs_table
    assert render_runs_table([]) == "No runs yet."


def test_render_runs_table_shows_fields():
    from dog_walker.main import render_runs_table
    from dog_walker.types import RunRecord
    rows = [RunRecord(
        session_id="1", provider="ollama", model="qwen2.5:7b", prompt="hello world",
        outcome="success", iterations=2, tool_calls=1, tools_used=["list_files"],
        latency_ms=1234, input_tokens=10, output_tokens=4, id=7,
        created_at="2026-07-08 10:00:00",
    )]
    out = render_runs_table(rows)
    assert "7" in out
    assert "success" in out
    assert "ollama" in out and "qwen2.5:7b" in out
    assert "hello world" in out
    assert "list_files" in out


def test_parse_limit_default_when_absent():
    from dog_walker.main import _parse_limit
    assert _parse_limit([]) == 20
    assert _parse_limit(["--limit"]) == 20           # missing value
    assert _parse_limit(["--limit", "abc"]) == 20    # non-integer, no crash
    assert _parse_limit(["--limit", "0"]) == 20      # non-positive
    assert _parse_limit(["--limit", "-3"]) == 20
    assert _parse_limit(["--limit", "5"]) == 5


def test_render_runs_table_handles_none_tokens():
    from dog_walker.main import render_runs_table
    from dog_walker.types import RunRecord
    rows = [RunRecord(
        session_id="1", provider="ollama", model="m", prompt="p",
        outcome="success", iterations=1, tool_calls=0, tools_used=[],
        latency_ms=3, input_tokens=None, output_tokens=None, id=1,
        created_at="2026-07-08 10:00:00",
    )]
    out = render_runs_table(rows)   # must not raise
    assert "success" in out
