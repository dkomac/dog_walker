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
