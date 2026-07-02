from __future__ import annotations
import sys
from middlemanager.config import Config, load_config
from middlemanager.storage.base import Storage
from middlemanager.storage.sqlite import SqliteStorage
from middlemanager.providers.base import Provider
from middlemanager.tools.builtin import build_registry
from middlemanager.loop import Harness


def build_storage(cfg: Config) -> Storage:
    if cfg.storage_backend == "sqlite":
        return SqliteStorage(cfg.storage_path)
    raise ValueError(f"Unknown storage backend: {cfg.storage_backend}")


def build_provider(cfg: Config) -> Provider:
    if cfg.provider_name == "anthropic":
        from middlemanager.providers.anthropic import AnthropicProvider
        return AnthropicProvider(cfg.model, cfg.max_tokens)
    if cfg.provider_name == "ollama":
        from middlemanager.providers.ollama import OllamaProvider
        return OllamaProvider(cfg.model)
    raise ValueError(f"Unknown provider: {cfg.provider_name}")


def build_harness(cfg: Config) -> Harness:
    provider = build_provider(cfg)
    registry = build_registry(cfg.enabled_tools, cfg.confirm_bash)
    storage = build_storage(cfg)
    return Harness(provider, registry, storage, cfg.max_iterations)


def main() -> None:
    cfg = load_config("config.toml")
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Usage: middlemanager <your prompt>")
        return
    harness = build_harness(cfg)
    try:
        result = harness.run(prompt)
    except Exception as e:
        print(f"Harness error: {e}")
        return
    print(result)


if __name__ == "__main__":
    main()
