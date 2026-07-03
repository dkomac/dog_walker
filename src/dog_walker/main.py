from __future__ import annotations
import os
import sys
from dog_walker.config import Config, load_config
from dog_walker.storage.base import Storage
from dog_walker.storage.sqlite import SqliteStorage
from dog_walker.providers.base import Provider
from dog_walker.tools.builtin import build_registry
from dog_walker.loop import Harness, build_system_prompt


def build_storage(cfg: Config) -> Storage:
    if cfg.storage_backend == "sqlite":
        return SqliteStorage(cfg.storage_path)
    raise ValueError(f"Unknown storage backend: {cfg.storage_backend}")


def build_provider(cfg: Config) -> Provider:
    if cfg.provider_name == "anthropic":
        from dog_walker.providers.anthropic import AnthropicProvider
        return AnthropicProvider(cfg.model, cfg.max_tokens)
    if cfg.provider_name == "ollama":
        from dog_walker.providers.ollama import OllamaProvider
        return OllamaProvider(cfg.model)
    raise ValueError(f"Unknown provider: {cfg.provider_name}")


def build_harness(cfg: Config) -> Harness:
    provider = build_provider(cfg)
    registry = build_registry(cfg.enabled_tools, cfg.confirm_bash)
    storage = build_storage(cfg)
    system_prompt = build_system_prompt(os.getcwd(), cfg.enabled_tools)
    return Harness(provider, registry, storage, cfg.max_iterations,
                   system_prompt=system_prompt)


def main() -> None:
    cfg = load_config("config.toml")
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Usage: dog_walker <your prompt>")
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
