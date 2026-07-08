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
