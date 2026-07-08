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


def _load_preferences(path: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def build_harness(cfg: Config, verbose: bool = False) -> Harness:
    provider = build_provider(cfg)
    registry = build_registry(cfg.enabled_tools, cfg.confirm_bash, cfg.preferences_file)
    storage = build_storage(cfg)
    preferences = _load_preferences(cfg.preferences_file)
    system_prompt = build_system_prompt(os.getcwd(), cfg.enabled_tools, preferences)
    return Harness(provider, registry, storage, cfg.max_iterations,
                   provider_name=cfg.provider_name, model=cfg.model,
                   system_prompt=system_prompt, verbose=verbose)


def main() -> None:
    cfg = load_config("config.toml")
    args = sys.argv[1:]
    verbose = False
    if "--verbose" in args or "-v" in args:
        verbose = True
        args = [a for a in args if a not in ("--verbose", "-v")]
    if args:
        prompt = " ".join(args)
    else:
        prompt = sys.stdin.read().strip()
    if not prompt:
        print("Usage: dog_walker [--verbose] <your prompt>")
        return
    harness = build_harness(cfg, verbose=verbose)
    try:
        result = harness.run(prompt)
    except Exception as e:
        print(f"Harness error: {e}")
        return
    print(result)


if __name__ == "__main__":
    main()
