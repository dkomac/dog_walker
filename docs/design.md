# Slim Agent Harness — Design

**Date:** 2026-07-02
**Status:** Approved, pre-implementation
**Goal:** Build a minimal but well-structured LLM agent harness in Python, as a learning exercise, to understand how modern agent harnesses (like Claude Code) work under the hood.

## Purpose & Scope

Build a "super slim" version of an LLM agent loop (the kind of thing that powers coding
agents). The primary goal is **learning**, not production use. The value is in seeing the
core loop clearly and understanding the two key abstraction boundaries that define a
well-built harness.

**In scope (v1):**
- A working agent loop against Anthropic (Claude).
- Four built-in tools: `read_file`, `write_file`, `list_files`, `run_bash`.
- Config via `config.toml`.
- Conversation persistence via SQLite.
- Two swappable abstractions: **Provider** (LLM backend) and **Storage** (database).

**Explicitly deferred (later steps):**
- Multi-agent / "middle manager" orchestration (sub-agents delegated work).
- Additional providers (OpenAI/GPT, local via Ollama) — the abstraction makes this a
  drop-in later.
- Postgres storage backend — again, a drop-in behind the Storage interface.

## Core Concept

A harness is a **loop**:

1. Send the model the conversation + the list of tools it may use.
2. The model replies with either a final answer (→ done) or a request to call tools.
3. If it requested tools, the harness runs them and appends the results to the
   conversation, then loops back to step 1.

Three principles that fall out of this:
- **The model never executes anything itself.** It only *requests* tool calls; the
  harness decides whether/how to run them. This is where permissions and safety live.
- **The message list is the entire memory.** The model is stateless between calls; every
  turn resends the whole conversation. "Context management" is just deciding what stays
  in that list.
- **The loop terminates exactly one way:** the model returns text with no tool calls.

## Architecture

```
                   ┌──────────────┐
   config.toml ───▶│              │
                   │   Harness    │◀──▶  Provider (LLM)   ── Anthropic adapter
   user input ────▶│    (loop)    │                          [GPT / Ollama later]
                   │              │
                   │              │◀──▶  Tool registry     ── read_file, write_file,
                   └──────────────┘                          list_files, run_bash
                          │
                          ▼
                    Storage (interface)  ── SQLite impl
                                            [Postgres later]
```

The central rule: `loop.py` depends only on the **interfaces** (`providers/base.py`,
`storage/base.py`, `tools/base.py`). It never imports `anthropic` or `sqlite3` directly.
That is what makes providers and storage swappable, and it is the single most important
idea in the project.

## Module Layout

Repository directory: `/Users/d/Desktop/git/middlemanger` (note: directory name is
misspelled; the Python package uses the correct spelling `middlemanager`).

```
middlemanger/
├── config.toml            # settings (exists)
├── data.db                # sqlite file (exists, empty)
├── pyproject.toml         # deps: anthropic, pytest (tomllib is stdlib in 3.11+)
├── src/middlemanager/
│   ├── __init__.py
│   ├── main.py            # entry point: load config, wire everything, run loop
│   ├── config.py          # parse config.toml → Config object
│   ├── loop.py            # THE HARNESS: the agent loop
│   ├── providers/
│   │   ├── base.py        # Provider interface (abstract)
│   │   └── anthropic.py   # Anthropic adapter
│   ├── tools/
│   │   ├── base.py        # Tool interface + registry
│   │   └── builtin.py     # read_file, write_file, list_files, run_bash
│   └── storage/
│       ├── base.py        # Storage interface (abstract)
│       └── sqlite.py      # SQLite implementation
└── tests/
```

## The Loop (pseudocode)

```
function run(user_input):
    messages = load_history_from_storage()      # or start fresh
    messages.append({role: "user", content: user_input})
    storage.save_message(user_message)

    for step in 1..max_iterations:               # safety cap from config
        response = provider.send(messages, tools)   # the API call

        storage.save_message(assistant_response)
        messages.append(assistant_response)

        if response has NO tool calls:
            return response.text                 # DONE

        for each tool_call in response.tool_calls:
            result = tool_registry.run(tool_call.name, tool_call.args)
            storage.save_tool_result(result)
            messages.append({role: "tool", content: result})
        # loop back — model now sees the tool results and continues

    return "stopped: hit max iterations"
```

## Interfaces

### Provider — `providers/base.py`
```python
class Provider(ABC):
    @abstractmethod
    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response:
        ...
```
The harness uses a **neutral internal shape** for messages, tool specs, and responses.
Each adapter translates neutral → provider JSON, calls the API, then translates the
response back → neutral. `Response` carries either final text or a list of tool calls.
Adding a new provider later = one new adapter; the loop is untouched.

### Tool — `tools/base.py`
```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict        # JSON schema describing how to call it

    @abstractmethod
    def run(self, **kwargs) -> str:
        ...
```
A registry (1) exposes tool specs to the model so it knows what it can call, and
(2) dispatches a call by name to the right `run()`. The `description` is how the model
decides when to use each tool, so it matters.

### Storage — `storage/base.py`
```python
class Storage(ABC):
    @abstractmethod
    def create_session(self) -> str: ...
    @abstractmethod
    def save_message(self, session_id, role, content): ...
    @abstractmethod
    def load_messages(self, session_id) -> list[Message]: ...
```
SQLite is the only place `sqlite3` is imported. A Postgres backend later implements the
same methods.

## Config Format (`config.toml`)

```toml
[provider]
name = "anthropic"          # which adapter to load
model = "claude-haiku-4-5-20251001"   # cheap + fast for a learning loop; swap to opus/sonnet anytime
max_tokens = 4096
# api_key comes from ANTHROPIC_API_KEY env var, NOT this file

[harness]
max_iterations = 20         # safety cap on the loop
confirm_bash = true         # y/n prompt before run_bash executes

[tools]
enabled = ["read_file", "write_file", "list_files", "run_bash"]

[storage]
backend = "sqlite"          # which storage impl to load
path = "data.db"
```

`provider.name`, `storage.backend`, and `tools.enabled` are strings the harness reads to
decide which classes to instantiate. This is how config drives the abstractions.

## Data Model (SQLite)

```
sessions(id TEXT PRIMARY KEY, created_at TEXT)

messages(
  id INTEGER PRIMARY KEY, session_id TEXT, seq INTEGER,
  role TEXT,           -- user | assistant | tool
  content TEXT,        -- JSON: handles text AND tool-calls/results uniformly
  created_at TEXT
)
```
Storing `content` as JSON lets one table shape hold plain text, tool-call requests, and
tool results without schema churn.

## Safety (v1)

`run_bash` can execute shell commands. For v1, when `harness.confirm_bash = true`, the
harness prints the command and requires a y/n confirmation before running it. This is
cheap and marks the exact spot where real harnesses put their permission system.

## Error Handling (v1)

- **API call fails** → catch, print the error, stop the loop cleanly.
- **Tool `run()` raises** → catch, feed the error text back to the model as the tool
  result. The model often recovers on its own (fixes a path, retries). Errors are data,
  not crashes.
- **`max_iterations` hit** → stop and report it.

## Testing Strategy

- **Tools** — unit-tested directly against temp files/dirs. No API needed.
- **Storage** — save→load round-trip tests against a temp SQLite db.
- **Loop** — tested with a **fake provider** implementing the `Provider` interface and
  returning scripted responses. Enables full loop testing with zero API cost. This is a
  primary payoff of the provider abstraction.
- **Provider adapter** — one small live smoke test against the real API, run manually.

## Future Directions (not in v1)

- Multi-agent orchestration: a coordinator ("middle manager") that spawns sub-agents for
  delegated subtasks and collects their results — built on top of this loop.
- Additional Provider adapters (OpenAI, Ollama).
- Postgres Storage backend.
