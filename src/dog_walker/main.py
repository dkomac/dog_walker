from __future__ import annotations
import os
import sys
from dog_walker.config import Config, load_config
from dog_walker.storage.base import Storage
from dog_walker.storage.sqlite import SqliteStorage
from dog_walker.providers.base import Provider
from dog_walker.tools.builtin import build_registry
from dog_walker.loop import Harness, build_system_prompt
from dog_walker.types import RunRecord


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


def _parse_limit(rest: list[str], default: int = 20) -> int:
    """Parse an optional `--limit N` from CLI args; fall back to default when
    it's missing, has no value, or isn't a positive integer."""
    if "--limit" not in rest:
        return default
    i = rest.index("--limit")
    if i + 1 >= len(rest):
        return default
    try:
        value = int(rest[i + 1])
    except ValueError:
        return default
    return value if value > 0 else default


def render_runs_table(runs: list[RunRecord]) -> str:
    if not runs:
        return "No runs yet."
    header = f"{'id':>3}  {'when':<19}  {'provider/model':<24}  {'outcome':<14}  {'it':>2}  {'tools':>5}  {'in/out tok':>12}  {'ms':>6}  prompt  tools_used"
    lines = [header]
    for r in runs:
        pm = f"{r.provider}/{r.model}"[:24]
        tok = f"{r.input_tokens if r.input_tokens is not None else '-'}/" \
              f"{r.output_tokens if r.output_tokens is not None else '-'}"
        prompt = (r.prompt or "").replace("\n", " ")
        if len(prompt) > 40:
            prompt = prompt[:39] + "…"
        when = (r.created_at or "")[:19]
        tools_used = ",".join(r.tools_used) if r.tools_used else ""
        lines.append(
            f"{r.id if r.id is not None else '-':>3}  {when:<19}  {pm:<24}  "
            f"{r.outcome:<14}  {r.iterations:>2}  {r.tool_calls:>5}  {tok:>12}  "
            f"{r.latency_ms:>6}  {prompt}  {tools_used}"
        )
    return "\n".join(lines)


def main() -> None:
    cfg = load_config("config.toml")
    args = sys.argv[1:]

    if args and args[0] == "runs":
        limit = _parse_limit(args[1:])
        storage = build_storage(cfg)
        print(render_runs_table(storage.list_runs(limit)))
        return

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
