# Runs Table — Design Spec

**Date:** 2026-07-08
**Status:** Approved, ready for implementation planning
**Sub-project 1 of the eval-loop track** (runs → eval suites → scoring → provider comparison → trace viewer)

## Goal

Instrument the harness so every run is recorded with enough metadata to later
evaluate and compare: model, prompt, tools used, iterations, latency, tokens,
and outcome. Make the data immediately visible via a `dog_walker runs` command.

This is the foundation the rest of the eval-loop track reads from. It is
valuable on its own: after this cycle you can see what each run did, how long it
took, and how many tokens it burned.

## What a "run" is

One `Harness.run(user_input)` call. A run spans multiple `provider.send()`
iterations; the loop accumulates metrics across those iterations and writes
**exactly one** `runs` row when the run ends (via any of its three exit paths).

## Scope

### In scope
- Capture token usage from both providers (Anthropic, Ollama).
- Accumulate per-run metrics in the loop and record one row per run.
- New `runs` table + `record_run` / `list_runs` on the `Storage` interface.
- `dog_walker runs [--limit N]` CLI command to print recent runs.
- Tests for all of the above (TDD).

### Out of scope (later cycles)
- Cost math / pricing tables (`cost_usd` column exists but stays NULL).
- Eval suites, task definitions, scoring, LLM-as-judge.
- Provider-comparison view, trace viewer.
- Sandboxing changes to `run_bash`.

## Data model — `runs` table

| column | type | notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `session_id` | TEXT | links to existing `sessions`/`messages` for the full transcript |
| `provider` | TEXT | e.g. `ollama`, `anthropic` |
| `model` | TEXT | e.g. `qwen2.5:7b` |
| `prompt` | TEXT | the user input |
| `final_answer` | TEXT NULL | the returned answer, if any |
| `outcome` | TEXT | `success` \| `max_iterations` \| `error` |
| `error` | TEXT NULL | exception message when `outcome=error` |
| `iterations` | INTEGER | number of loop steps that ran |
| `tool_calls` | INTEGER | total tool calls across the run |
| `tools_used` | TEXT | JSON array of **distinct** tool names, in first-seen order |
| `input_tokens` | INTEGER NULL | summed across iterations; NULL if provider reported nothing |
| `output_tokens` | INTEGER NULL | summed across iterations; NULL if provider reported nothing |
| `cost_usd` | REAL NULL | reserved for a later cycle; always NULL now |
| `latency_ms` | INTEGER | wall-clock of the whole run |
| `created_at` | TEXT | `DEFAULT CURRENT_TIMESTAMP` |

Token-sum nullability rule: the None-vs-sum decision is made **per field independently**. A field stays NULL until some iteration reports a non-NULL value for that specific field; once a value is reported, subsequent NULL values are treated as 0 when summing. This allows storing NULL for genuinely-unknown counts (more honest than asserting 0).

## Types (`types.py`)

Add a `Usage` dataclass and extend `Response`:

```python
@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None

@dataclass
class Response:
    text: str | None
    tool_calls: list[ToolCall]
    usage: Usage | None = None   # default keeps fake provider + existing tests intact
```

Add a `RunRecord` dataclass to keep the `Storage` interface clean. It carries
the writable fields; `id` and `created_at` are assigned by storage on write and
populated on read:

```python
@dataclass
class RunRecord:
    session_id: str
    provider: str
    model: str
    prompt: str
    outcome: str
    iterations: int
    tool_calls: int
    tools_used: list[str]
    latency_ms: int
    final_answer: str | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    id: int | None = None
    created_at: str | None = None
```

## Capturing usage in providers

The usage data is already returned by both APIs and currently discarded.

- **Anthropic** (`providers/anthropic.py`): `resp.usage.input_tokens` and
  `resp.usage.output_tokens`. Thread a `Usage` into `parse_anthropic_response`
  (add a `usage` parameter) and return it on the `Response`.
- **Ollama** (`providers/ollama.py`): `resp.prompt_eval_count` (input) and
  `resp.eval_count` (output). These may be absent/None on some responses — read
  defensively with `getattr(..., None)`. Thread into `parse_ollama_response`.
- **Fake** (`providers/fake.py`): unchanged. It returns scripted `Response`s;
  tests that care about usage set it explicitly on the scripted response.

## Instrumenting the loop (`loop.py`)

`Harness.run()` needs the harness to know `provider` and `model` for the record.
Both are available on the provider instances (`self.provider.model`; provider
name is derivable). Simplest: add a `provider_name: str` and read `model` off the
provider, or pass both into `Harness.__init__` from `build_harness`. **Decision:**
pass `provider_name` and `model` into `Harness.__init__` (wired in `main.py`
from `cfg.provider_name` / `cfg.model`) — avoids reaching into provider
internals and keeps the loop provider-agnostic.

Inside `run()`:
- Start a wall-clock timer at the top.
- Accumulate across iterations: `input_tokens`, `output_tokens` (None-aware per
  the nullability rule), total `tool_calls`, and distinct `tools_used` (ordered).
- Track `iterations` = the step count reached.
- Determine `outcome`:
  - returned a final answer (no tool calls) → `success`
  - loop exhausted `max_iterations` → `max_iterations`
  - exception raised anywhere in the loop → `error` (capture `str(e)`)
- Use a `try/finally` so the run is recorded on **every** exit path. On
  exception: record with `outcome=error`, then **re-raise** so `main.py`'s
  existing `except Exception` still prints `Harness error: ...`.

The existing behavior (return final answer string, or the max-iterations
message) is preserved; recording is additive.

## Storage (`storage/base.py`, `storage/sqlite.py`)

Extend the `Storage` ABC:

```python
@abstractmethod
def record_run(self, run: RunRecord) -> str: ...   # returns new run id

@abstractmethod
def list_runs(self, limit: int = 20) -> list[RunRecord]: ...
```

`SqliteStorage`:
- `CREATE TABLE IF NOT EXISTS runs (...)` in `__init__` (matches existing
  sessions/messages creation style).
- `record_run`: INSERT the writable columns (`tools_used` serialized with
  `json.dumps`), commit, return `str(lastrowid)`.
- `list_runs`: `SELECT ... ORDER BY id DESC LIMIT ?`, deserialize `tools_used`
  with `json.loads`, return `RunRecord`s with `id`/`created_at` populated.

Existing DBs: the `CREATE TABLE IF NOT EXISTS` runs on every open, so old
`data.db` files gain the table automatically. No migration needed.

## CLI — `dog_walker runs` (`main.py`)

`main()` currently treats all args as the prompt. Add a subcommand check
**before** prompt handling:

- `dog_walker runs [--limit N]` → build storage from config, call
  `list_runs(limit)` (default 20), print a table and return. Does **not** invoke
  the model.
- Everything else behaves exactly as today.

Table columns (truncate long fields to keep rows on one line):
`id · time · provider/model · outcome · iters · tools · in/out tok · latency(ms) · prompt`

Keep formatting simple and dependency-free (plain string formatting / f-strings).
Empty state: print a friendly "No runs yet." when the table is empty.

## Testing (TDD — write tests first)

All against the `fake` provider / real `SqliteStorage` on a tmp path.

1. **types**: `Response(...)` defaults `usage=None`; `Response` accepts a
   `Usage`. `RunRecord` defaults are correct.
2. **anthropic parser**: given a stub response object exposing
   `usage.input_tokens/output_tokens`, `parse_anthropic_response` returns a
   `Response` whose `usage` matches.
3. **ollama parser**: given a stub message/response with
   `prompt_eval_count`/`eval_count`, `parse_ollama_response` populates `usage`;
   given a response missing those, `usage` tokens are None (no crash).
4. **storage**: `record_run` then `list_runs` round-trips every field;
   `tools_used` survives JSON serialization; ordering is most-recent-first;
   `limit` is honored.
5. **loop — success**: fake provider returns a final answer with `usage`; assert
   one run recorded with `outcome=success`, correct token sums, `tool_calls=0`,
   `iterations=1`, and `final_answer` set.
6. **loop — tokens summed + tools collected**: fake scripts one tool call then a
   final answer; assert `tool_calls`/`tools_used`/`iterations` and summed tokens
   across the two `send()`s.
7. **loop — max_iterations**: fake always returns a tool call; assert
   `outcome=max_iterations` and `iterations == max_iterations`.
8. **loop — error**: fake (or a stub) raises during `send`; assert a run is
   recorded with `outcome=error` and non-null `error`, **and** the exception
   propagates.
9. **CLI smoke**: after recording a couple of runs, the `runs` command prints
   their ids/outcomes; empty DB prints the empty-state message.

## Files touched

- `src/dog_walker/types.py` — `Usage`, `Response.usage`, `RunRecord`
- `src/dog_walker/providers/anthropic.py` — capture usage
- `src/dog_walker/providers/ollama.py` — capture usage
- `src/dog_walker/loop.py` — accumulate metrics, record run, `Harness.__init__`
  gains `provider_name`/`model`
- `src/dog_walker/storage/base.py` — `record_run` / `list_runs`
- `src/dog_walker/storage/sqlite.py` — `runs` table + methods
- `src/dog_walker/main.py` — `runs` subcommand, wire provider_name/model into
  `build_harness`/`Harness`
- `tests/` — new tests per the list above; update `test_loop.py` /
  `test_wiring.py` construction of `Harness` for the new `__init__` params
- `README.md` — brief mention of `dog_walker runs`
```
