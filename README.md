# dog_walker

A tiny LLM **agent harness**, built for learning. It runs a model in a loop: the model
calls tools (read/write files, list directories, run shell commands), your code executes
them and feeds the results back, and it repeats until the model gives a final answer.

Model providers (Anthropic or local Ollama) and storage (SQLite) are swappable behind
small interfaces.

## Requirements

- Python 3.11+
- Either a local [Ollama](https://ollama.com) install, or an Anthropic API key

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configure

Pick a provider in `config.toml`:

```toml
[provider]
name = "ollama"            # or "anthropic"
model = "llama3.1:8b"      # or e.g. "claude-haiku-4-5-20251001"
```

- **Ollama:** make sure it's running and the model is pulled (`ollama pull llama3.1:8b`).
- **Anthropic:** export your key — `export ANTHROPIC_API_KEY="sk-ant-..."`.

## Run

```bash
dog_walker "list the files here and tell me what this project is"
dog_walker --verbose "read config.toml and tell me which model is set"
```

`--verbose` traces each tool call and result live. Conversations persist to `data.db`.

## Tests

```bash
pytest
```

## How it works

- [docs/design.md](docs/design.md) — the architecture and why it's shaped this way
- [docs/implementation-plan.md](docs/implementation-plan.md) — the step-by-step build
