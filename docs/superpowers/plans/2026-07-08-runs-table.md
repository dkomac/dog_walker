# Runs Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record one metrics row per harness run (model, prompt, tools, iterations, latency, tokens, outcome) and expose it via a `dog_walker runs` command.

**Architecture:** Extend `Response` to carry token `Usage` captured from each provider. The loop accumulates usage/latency/tool-metrics across its iterations and, on every exit path, writes one `RunRecord` to a new SQLite `runs` table through two new `Storage` methods. A `dog_walker runs` subcommand reads that table back and prints it.

**Tech Stack:** Python 3.11+, stdlib `sqlite3`, `json`, `time`, `dataclasses`; `pytest` for tests. No new dependencies.

## Global Constraints

- Python 3.11+; use `from __future__ import annotations` at the top of every source file (matches existing files).
- No new third-party dependencies.
- TDD: write the failing test first, watch it fail, implement, watch it pass, commit.
- `cost_usd` exists as a nullable column but is **always NULL** this cycle — no pricing logic.
- Recording is **additive**: existing return values and error-printing behavior must be preserved.
- Token nullability rule: if **no** iteration reported usage, store `None` for both token columns; if **any** did, store the sum treating missing per-iteration values as 0.
- Follow existing code style: small modules, dataclasses in `types.py`, `CREATE TABLE IF NOT EXISTS` in `SqliteStorage.__init__`.

---

## File Structure

- `src/dog_walker/types.py` — add `Usage`, `Response.usage` field, `RunRecord` dataclass.
- `src/dog_walker/providers/anthropic.py` — capture usage from `resp.usage`.
- `src/dog_walker/providers/ollama.py` — capture usage from `resp.prompt_eval_count`/`eval_count`.
- `src/dog_walker/storage/base.py` — add `record_run` / `list_runs` abstract methods.
- `src/dog_walker/storage/sqlite.py` — `runs` table + `record_run` / `list_runs`.
- `src/dog_walker/loop.py` — accumulate metrics; `Harness.__init__` gains `provider_name` + `model`; record run in `finally`.
- `src/dog_walker/main.py` — `runs` subcommand; pass `provider_name`/`model` into `Harness`.
- `tests/` — new tests; update existing `Harness(...)` construction sites.
- `README.md` — mention `dog_walker runs`.

Tasks are ordered so each builds only on earlier ones (types → providers → storage → loop → CLI → docs).

---

### Task 1: Types — `Usage`, `Response.usage`, `RunRecord`

**Files:**
- Modify: `src/dog_walker/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Usage(input_tokens: int | None = None, output_tokens: int | None = None)`
  - `Response(text, tool_calls, usage: Usage | None = None)`
  - `RunRecord` dataclass with fields (all keyword-constructible):
    `session_id: str, provider: str, model: str, prompt: str, outcome: str, iterations: int, tool_calls: int, tools_used: list[str], latency_ms: int, final_answer: str | None = None, error: str | None = None, input_tokens: int | None = None, output_tokens: int | None = None, cost_usd: float | None = None, id: int | None = None, created_at: str | None = None`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_types.py`:

```python
from dog_walker.types import Usage, Response, RunRecord


def test_response_usage_defaults_none():
    r = Response(text="hi", tool_calls=[])
    assert r.usage is None


def test_response_carries_usage():
    r = Response(text="hi", tool_calls=[], usage=Usage(input_tokens=10, output_tokens=3))
    assert r.usage.input_tokens == 10
    assert r.usage.output_tokens == 3


def test_usage_defaults_none():
    u = Usage()
    assert u.input_tokens is None and u.output_tokens is None


def test_runrecord_minimal_and_defaults():
    rec = RunRecord(
        session_id="1", provider="fake", model="m", prompt="p",
        outcome="success", iterations=1, tool_calls=0, tools_used=[],
        latency_ms=5,
    )
    assert rec.final_answer is None
    assert rec.error is None
    assert rec.input_tokens is None and rec.output_tokens is None
    assert rec.cost_usd is None
    assert rec.id is None and rec.created_at is None
    assert rec.tools_used == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ImportError: cannot import name 'Usage'` (or `RunRecord`).

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/types.py`, add `Usage` above `Response`, add the `usage` field to `Response`, and add `RunRecord`:

```python
@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Response:
    text: str | None
    tool_calls: list[ToolCall]
    usage: Usage | None = None


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

(Replace the existing `Response` definition; keep `field`/`asdict` imports intact.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_types.py -v`
Expected: PASS (all four new tests, plus existing ones still green).

- [ ] **Step 5: Commit**

```bash
git add src/dog_walker/types.py tests/test_types.py
git commit -m "feat: add Usage and RunRecord types, Response.usage field"
```

---

### Task 2: Anthropic provider captures token usage

**Files:**
- Modify: `src/dog_walker/providers/anthropic.py`
- Test: `tests/test_anthropic_adapter.py`

**Interfaces:**
- Consumes: `Usage`, `Response` (Task 1).
- Produces: `parse_anthropic_response(content_blocks, usage: Usage | None = None) -> Response` — attaches `usage` to the returned `Response`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_anthropic_adapter.py`:

```python
from types import SimpleNamespace
from dog_walker.providers.anthropic import parse_anthropic_response
from dog_walker.types import Usage


def test_parse_attaches_usage():
    blocks = [SimpleNamespace(type="text", text="done")]
    usage = Usage(input_tokens=42, output_tokens=7)
    resp = parse_anthropic_response(blocks, usage)
    assert resp.text == "done"
    assert resp.usage.input_tokens == 42
    assert resp.usage.output_tokens == 7


def test_parse_without_usage_is_none():
    blocks = [SimpleNamespace(type="text", text="done")]
    resp = parse_anthropic_response(blocks)
    assert resp.usage is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anthropic_adapter.py -v`
Expected: FAIL — `parse_anthropic_response()` takes 1 positional arg but 2 given (or `usage` attribute missing).

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/providers/anthropic.py`:

Change the parser signature and return:

```python
def parse_anthropic_response(content_blocks: list, usage: Usage | None = None) -> Response:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))
    text = "\n".join(text_parts) if text_parts else None
    return Response(text=text, tool_calls=tool_calls, usage=usage)
```

Update the import line to include `Usage`:

```python
from dog_walker.types import Message, ToolSpec, Response, ToolCall, Usage
```

In `AnthropicProvider.send`, build `Usage` from the SDK response and pass it:

```python
        resp = self.client.messages.create(**kwargs)
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = Usage(
                input_tokens=getattr(resp.usage, "input_tokens", None),
                output_tokens=getattr(resp.usage, "output_tokens", None),
            )
        return parse_anthropic_response(resp.content, usage)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_anthropic_adapter.py -v`
Expected: PASS (new + existing tests).

- [ ] **Step 5: Commit**

```bash
git add src/dog_walker/providers/anthropic.py tests/test_anthropic_adapter.py
git commit -m "feat: capture token usage from Anthropic responses"
```

---

### Task 3: Ollama provider captures token usage

**Files:**
- Modify: `src/dog_walker/providers/ollama.py`
- Test: `tests/test_ollama_adapter.py`

**Interfaces:**
- Consumes: `Usage`, `Response` (Task 1).
- Produces: `parse_ollama_response(message, usage: Usage | None = None) -> Response`. Provider reads `prompt_eval_count`/`eval_count` off the top-level chat response (not the message) defensively.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ollama_adapter.py`:

```python
from types import SimpleNamespace
from dog_walker.providers.ollama import parse_ollama_response
from dog_walker.types import Usage


def test_parse_attaches_usage():
    msg = SimpleNamespace(content="done", tool_calls=None)
    resp = parse_ollama_response(msg, Usage(input_tokens=11, output_tokens=5))
    assert resp.text == "done"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 5


def test_parse_without_usage_is_none():
    msg = SimpleNamespace(content="done", tool_calls=None)
    resp = parse_ollama_response(msg)
    assert resp.usage is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ollama_adapter.py -v`
Expected: FAIL — `parse_ollama_response()` takes 1 positional arg but 2 given.

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/providers/ollama.py`:

Update the import to include `Usage`:

```python
from dog_walker.types import Message, ToolSpec, Response, ToolCall, Usage
```

Change the parser signature and return line:

```python
def parse_ollama_response(message, usage: Usage | None = None) -> Response:
    text = getattr(message, "content", "") or None
    tool_calls: list[ToolCall] = []
    raw = getattr(message, "tool_calls", None) or []
    for i, tc in enumerate(raw):
        fn = tc.function
        tool_calls.append(ToolCall(id=f"call_{i}", name=fn.name, args=dict(fn.arguments)))
    return Response(text=text, tool_calls=tool_calls, usage=usage)
```

In `OllamaProvider.send`, read counts off the response defensively and pass a `Usage`:

```python
        resp = self.client.chat(
            model=self.model,
            messages=to_ollama_messages(messages),
            tools=to_ollama_tools(tools),
            stream=False,
            options={"temperature": self.temperature},
        )
        in_tok = getattr(resp, "prompt_eval_count", None)
        out_tok = getattr(resp, "eval_count", None)
        usage = None
        if in_tok is not None or out_tok is not None:
            usage = Usage(input_tokens=in_tok, output_tokens=out_tok)
        return parse_ollama_response(resp.message, usage)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ollama_adapter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dog_walker/providers/ollama.py tests/test_ollama_adapter.py
git commit -m "feat: capture token usage from Ollama responses"
```

---

### Task 4: Storage — `runs` table, `record_run`, `list_runs`

**Files:**
- Modify: `src/dog_walker/storage/base.py`
- Modify: `src/dog_walker/storage/sqlite.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `RunRecord` (Task 1).
- Produces:
  - `Storage.record_run(self, run: RunRecord) -> str` — persists, returns new id as string.
  - `Storage.list_runs(self, limit: int = 20) -> list[RunRecord]` — most-recent-first, each `RunRecord` has `id` and `created_at` populated, `tools_used` deserialized.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_storage.py`:

```python
from dog_walker.storage.sqlite import SqliteStorage
from dog_walker.types import RunRecord


def _rec(**over):
    base = dict(
        session_id="1", provider="fake", model="m", prompt="p",
        outcome="success", iterations=1, tool_calls=0, tools_used=[],
        latency_ms=5,
    )
    base.update(over)
    return RunRecord(**base)


def test_record_and_list_roundtrip(tmp_path):
    s = SqliteStorage(str(tmp_path / "t.db"))
    rid = s.record_run(_rec(
        prompt="hello", tools_used=["list_files", "run_bash"],
        input_tokens=10, output_tokens=4, tool_calls=2,
    ))
    assert rid is not None
    runs = s.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.prompt == "hello"
    assert r.tools_used == ["list_files", "run_bash"]
    assert r.input_tokens == 10 and r.output_tokens == 4
    assert r.tool_calls == 2
    assert r.id is not None
    assert r.created_at is not None
    assert r.cost_usd is None


def test_list_runs_most_recent_first_and_limit(tmp_path):
    s = SqliteStorage(str(tmp_path / "t.db"))
    for i in range(3):
        s.record_run(_rec(prompt=f"p{i}"))
    runs = s.list_runs(limit=2)
    assert len(runs) == 2
    assert runs[0].prompt == "p2"   # newest first
    assert runs[1].prompt == "p1"


def test_null_tokens_roundtrip(tmp_path):
    s = SqliteStorage(str(tmp_path / "t.db"))
    s.record_run(_rec(input_tokens=None, output_tokens=None))
    r = s.list_runs()[0]
    assert r.input_tokens is None and r.output_tokens is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL — `AttributeError: 'SqliteStorage' object has no attribute 'record_run'`.

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/storage/base.py`, add the abstract methods and import:

```python
from dog_walker.types import Message, RunRecord
```
```python
    @abstractmethod
    def record_run(self, run: RunRecord) -> str: ...

    @abstractmethod
    def list_runs(self, limit: int = 20) -> list[RunRecord]: ...
```

In `src/dog_walker/storage/sqlite.py`, import `RunRecord`:

```python
from dog_walker.types import Message, RunRecord
```

Add the table creation in `__init__` (after the `messages` table, before `self.conn.commit()`):

```python
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT,"
            " provider TEXT,"
            " model TEXT,"
            " prompt TEXT,"
            " final_answer TEXT,"
            " outcome TEXT,"
            " error TEXT,"
            " iterations INTEGER,"
            " tool_calls INTEGER,"
            " tools_used TEXT,"
            " input_tokens INTEGER,"
            " output_tokens INTEGER,"
            " cost_usd REAL,"
            " latency_ms INTEGER,"
            " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
```

Add the two methods:

```python
    def record_run(self, run: RunRecord) -> str:
        cur = self.conn.execute(
            "INSERT INTO runs (session_id, provider, model, prompt, final_answer,"
            " outcome, error, iterations, tool_calls, tools_used, input_tokens,"
            " output_tokens, cost_usd, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run.session_id, run.provider, run.model, run.prompt, run.final_answer,
             run.outcome, run.error, run.iterations, run.tool_calls,
             json.dumps(run.tools_used), run.input_tokens, run.output_tokens,
             run.cost_usd, run.latency_ms),
        )
        self.conn.commit()
        return str(cur.lastrowid)

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        rows = self.conn.execute(
            "SELECT id, session_id, provider, model, prompt, final_answer, outcome,"
            " error, iterations, tool_calls, tools_used, input_tokens, output_tokens,"
            " cost_usd, latency_ms, created_at"
            " FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        out: list[RunRecord] = []
        for r in rows:
            out.append(RunRecord(
                id=r[0], session_id=r[1], provider=r[2], model=r[3], prompt=r[4],
                final_answer=r[5], outcome=r[6], error=r[7], iterations=r[8],
                tool_calls=r[9], tools_used=json.loads(r[10]) if r[10] else [],
                input_tokens=r[11], output_tokens=r[12], cost_usd=r[13],
                latency_ms=r[14], created_at=r[15],
            ))
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dog_walker/storage/base.py src/dog_walker/storage/sqlite.py tests/test_storage.py
git commit -m "feat: add runs table with record_run and list_runs"
```

---

### Task 5: Loop records a run per `Harness.run()`

**Files:**
- Modify: `src/dog_walker/loop.py`
- Modify: `tests/test_loop.py` (existing `Harness(...)` calls gain new params)
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `RunRecord` (Task 1), `Storage.record_run` (Task 4), `Response.usage` (Task 1).
- Produces: `Harness.__init__(self, provider, registry, storage, max_iterations, provider_name: str = "", model: str = "", system_prompt=DEFAULT_SYSTEM_PROMPT, verbose=False)` — records exactly one `RunRecord` per `run()` call on every exit path.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_loop.py` (uses the existing `_storage`, `FakeProvider`, `reg` helpers already in that file; `Response`, `ToolCall`, `Usage` from `dog_walker.types`):

```python
from dog_walker.types import Usage


def test_records_success_run(tmp_path):
    storage = _storage(tmp_path)
    provider = FakeProvider([
        Response(text="the answer", tool_calls=[], usage=Usage(input_tokens=8, output_tokens=2)),
    ])
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    harness.run("do it")
    runs = storage.list_runs()
    assert len(runs) == 1
    r = runs[0]
    assert r.outcome == "success"
    assert r.prompt == "do it"
    assert r.final_answer == "the answer"
    assert r.iterations == 1
    assert r.tool_calls == 0
    assert r.input_tokens == 8 and r.output_tokens == 2
    assert r.provider == "fake" and r.model == "m"
    assert r.latency_ms >= 0


def test_records_tokens_summed_and_tools_collected(tmp_path):
    storage = _storage(tmp_path)
    provider = FakeProvider([
        Response(text=None,
                 tool_calls=[ToolCall(id="c0", name="list_files", args={"path": "."})],
                 usage=Usage(input_tokens=5, output_tokens=1)),
        Response(text="done", tool_calls=[], usage=Usage(input_tokens=6, output_tokens=3)),
    ])
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    harness.run("list")
    r = storage.list_runs()[0]
    assert r.outcome == "success"
    assert r.iterations == 2
    assert r.tool_calls == 1
    assert r.tools_used == ["list_files"]
    assert r.input_tokens == 11 and r.output_tokens == 4


def test_records_max_iterations(tmp_path):
    storage = _storage(tmp_path)
    looping = [
        Response(text=None,
                 tool_calls=[ToolCall(id="c0", name="list_files", args={"path": "."})])
        for _ in range(10)
    ]
    provider = FakeProvider(looping)
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=3,
                      provider_name="fake", model="m")
    harness.run("loop forever")
    r = storage.list_runs()[0]
    assert r.outcome == "max_iterations"
    assert r.iterations == 3


def test_no_usage_reported_stores_null_tokens(tmp_path):
    storage = _storage(tmp_path)
    provider = FakeProvider([Response(text="hi", tool_calls=[])])
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(provider, reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    harness.run("hi")
    r = storage.list_runs()[0]
    assert r.input_tokens is None and r.output_tokens is None


def test_records_error_and_reraises(tmp_path):
    import pytest

    class BoomProvider:
        model = "m"
        def send(self, messages, tools):
            raise RuntimeError("boom")

    storage = _storage(tmp_path)
    reg = build_registry(["list_files"], confirm_bash=False)
    harness = Harness(BoomProvider(), reg, storage, max_iterations=5,
                      provider_name="fake", model="m")
    with pytest.raises(RuntimeError, match="boom"):
        harness.run("go")
    r = storage.list_runs()[0]
    assert r.outcome == "error"
    assert "boom" in r.error
```

Note: this test file already imports `Response`, `ToolCall`, `build_registry`, `Harness`, `FakeProvider` and defines `_storage`. Only add what's missing (`Usage` import, and `import pytest` is inside the one test). Verify the existing `build_registry` import/signature in the file and match it; if the existing tests call `build_registry(reg_args...)` differently, mirror their exact call.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL — `Harness.__init__() got an unexpected keyword argument 'provider_name'` (and no runs recorded).

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/loop.py`:

Add imports at the top (join the existing `types` import):

```python
import time
from dog_walker.types import Message, ToolResult, RunRecord
```

Extend `Harness.__init__` to accept and store `provider_name` and `model`:

```python
    def __init__(self, provider: Provider, registry: ToolRegistry,
                 storage: Storage, max_iterations: int,
                 provider_name: str = "", model: str = "",
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 verbose: bool = False):
        self.provider = provider
        self.registry = registry
        self.storage = storage
        self.max_iterations = max_iterations
        self.provider_name = provider_name
        self.model = model
        self.system_prompt = system_prompt
        self.verbose = verbose
```

Rewrite `run()` to accumulate metrics and record in a `finally`:

```python
    def run(self, user_input: str) -> str:
        session_id = self.storage.create_session()
        messages: list[Message] = []

        if self.system_prompt:
            system_msg = Message(role="system", text=self.system_prompt)
            messages.append(system_msg)
            self.storage.save_message(session_id, system_msg)

        user_msg = Message(role="user", text=user_input)
        messages.append(user_msg)
        self.storage.save_message(session_id, user_msg)

        start = time.monotonic()
        iterations = 0
        tool_calls = 0
        tools_used: list[str] = []
        in_tokens: int | None = None
        out_tokens: int | None = None
        outcome = "max_iterations"
        final_answer: str | None = None
        error: str | None = None

        def _add_tokens(usage):
            nonlocal in_tokens, out_tokens
            if usage is None:
                return
            if usage.input_tokens is not None:
                in_tokens = (in_tokens or 0) + usage.input_tokens
            if usage.output_tokens is not None:
                out_tokens = (out_tokens or 0) + usage.output_tokens

        try:
            for step in range(1, self.max_iterations + 1):
                iterations = step
                self._trace(f"\n── iteration {step} ──")
                response = self.provider.send(messages, self.registry.specs())
                _add_tokens(response.usage)

                assistant_msg = Message(role="assistant", text=response.text,
                                        tool_calls=response.tool_calls)
                messages.append(assistant_msg)
                self.storage.save_message(session_id, assistant_msg)

                if not response.tool_calls:
                    self._trace("💬 final answer (no tool calls) → stopping")
                    outcome = "success"
                    final_answer = response.text or ""
                    return final_answer

                results = []
                for call in response.tool_calls:
                    tool_calls += 1
                    if call.name not in tools_used:
                        tools_used.append(call.name)
                    self._trace(f"→ tool call: {call.name}({_fmt_args(call.args)})")
                    output = self.registry.run(call.name, call.args)
                    self._trace(f"← result: {_truncate(output)}")
                    results.append(ToolResult(tool_call_id=call.id, content=output))

                tool_msg = Message(role="tool", tool_results=results)
                messages.append(tool_msg)
                self.storage.save_message(session_id, tool_msg)

            return f"Stopped: reached max iterations ({self.max_iterations})."
        except Exception as e:
            outcome = "error"
            error = str(e)
            raise
        finally:
            latency_ms = int((time.monotonic() - start) * 1000)
            self.storage.record_run(RunRecord(
                session_id=session_id,
                provider=self.provider_name,
                model=self.model,
                prompt=user_input,
                outcome=outcome,
                iterations=iterations,
                tool_calls=tool_calls,
                tools_used=tools_used,
                latency_ms=latency_ms,
                final_answer=final_answer,
                error=error,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop.py -v`
Expected: PASS. Then run the whole suite: `pytest -v` — fix any existing `Harness(...)` construction in `tests/test_wiring.py` that breaks (the new params are keyword-optional, so existing positional calls should still work; only fix if a test asserts on run recording).

- [ ] **Step 5: Commit**

```bash
git add src/dog_walker/loop.py tests/test_loop.py
git commit -m "feat: record a RunRecord per harness run on every exit path"
```

---

### Task 6: Wire provider name/model into the harness (`main.py`)

**Files:**
- Modify: `src/dog_walker/main.py`
- Test: `tests/test_wiring.py`

**Interfaces:**
- Consumes: `Harness.__init__(..., provider_name, model, ...)` (Task 5), `cfg.provider_name`, `cfg.model` (existing `Config`).
- Produces: `build_harness` constructs `Harness` with `provider_name=cfg.provider_name, model=cfg.model`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_wiring.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wiring.py::test_build_harness_passes_provider_and_model -v`
Expected: FAIL — `harness.provider_name` is `""` (default), not `"ollama"`.

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/main.py`, update the `Harness(...)` construction in `build_harness`:

```python
    return Harness(provider, registry, storage, cfg.max_iterations,
                   provider_name=cfg.provider_name, model=cfg.model,
                   system_prompt=system_prompt, verbose=verbose)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dog_walker/main.py tests/test_wiring.py
git commit -m "feat: wire provider name and model into Harness"
```

---

### Task 7: `dog_walker runs` command

**Files:**
- Modify: `src/dog_walker/main.py`
- Test: `tests/test_wiring.py`

**Interfaces:**
- Consumes: `Storage.list_runs` (Task 4), `build_storage` (existing), `load_config` (existing).
- Produces: `render_runs_table(runs: list[RunRecord]) -> str` (pure, testable) and a `runs` branch in `main()` that prints it. Empty list → `"No runs yet."`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_wiring.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_wiring.py -v`
Expected: FAIL — `cannot import name 'render_runs_table'`.

- [ ] **Step 3: Write minimal implementation**

In `src/dog_walker/main.py`, add a renderer and a `runs` branch. Add `RunRecord` to imports if needed for typing (not required at runtime). Renderer:

```python
def render_runs_table(runs: list) -> str:
    if not runs:
        return "No runs yet."
    header = f"{'id':>3}  {'when':<19}  {'provider/model':<24}  {'outcome':<14}  {'it':>2}  {'tools':>5}  {'in/out tok':>12}  {'ms':>6}  prompt"
    lines = [header]
    for r in runs:
        pm = f"{r.provider}/{r.model}"[:24]
        tok = f"{r.input_tokens if r.input_tokens is not None else '-'}/" \
              f"{r.output_tokens if r.output_tokens is not None else '-'}"
        prompt = (r.prompt or "").replace("\n", " ")
        if len(prompt) > 40:
            prompt = prompt[:39] + "…"
        when = (r.created_at or "")[:19]
        lines.append(
            f"{r.id if r.id is not None else '-':>3}  {when:<19}  {pm:<24}  "
            f"{r.outcome:<14}  {r.iterations:>2}  {r.tool_calls:>5}  {tok:>12}  "
            f"{r.latency_ms:>6}  {prompt}"
        )
    return "\n".join(lines)
```

In `main()`, add the subcommand branch **before** the prompt handling (right after `args = sys.argv[1:]`, before the verbose parsing — or after verbose parsing but before building the harness; place it right after computing `args`):

```python
def main() -> None:
    cfg = load_config("config.toml")
    args = sys.argv[1:]

    if args and args[0] == "runs":
        limit = 20
        rest = args[1:]
        if "--limit" in rest:
            i = rest.index("--limit")
            if i + 1 < len(rest):
                limit = int(rest[i + 1])
        storage = build_storage(cfg)
        print(render_runs_table(storage.list_runs(limit)))
        return

    verbose = False
    ...
```

(Keep the rest of `main()` unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_wiring.py -v`
Expected: PASS. Then full suite: `pytest -v` — all green.

- [ ] **Step 5: Manual smoke check**

```bash
python -c "import subprocess" # sanity
dog_walker runs            # prints "No runs yet." on a fresh db, or the table
```
Expected: a table (or "No runs yet.") — no model call, no crash.

- [ ] **Step 6: Commit**

```bash
git add src/dog_walker/main.py tests/test_wiring.py
git commit -m "feat: add dog_walker runs command"
```

---

### Task 8: README — document `dog_walker runs`

**Files:**
- Modify: `README.md`
- Test: none (docs).

- [ ] **Step 1: Add a section**

Under the `## Run` section (after the `--verbose` line), add:

```markdown
Every run is recorded (model, prompt, tools used, iterations, latency, tokens,
outcome). List recent runs:

```bash
dog_walker runs            # last 20 runs
dog_walker runs --limit 5  # last 5
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document dog_walker runs command"
```

---

## Self-Review Notes

- **Spec coverage:** runs table (Task 4) ✓; token capture Anthropic (Task 2) + Ollama (Task 3) ✓; loop accumulation + 3 outcomes + finally recording (Task 5) ✓; `record_run`/`list_runs` interface (Task 4) ✓; `RunRecord`/`Usage`/`Response.usage` (Task 1) ✓; provider/model wiring (Task 6) ✓; `runs` CLI + empty state (Task 7) ✓; README (Task 8) ✓. Cost stays NULL (never written) ✓. Token nullability rule implemented in `_add_tokens` + Task 5 tests ✓.
- **Type consistency:** `RunRecord` field names identical across Tasks 1/4/5/7; `parse_*_response(blocks/message, usage=None)` signatures consistent Tasks 2/3; `Harness.__init__` signature consistent Tasks 5/6.
- **No placeholders:** every code step shows full code; every test step shows full assertions.
- **Note for implementer:** Tasks 5's tests reference `build_registry` and `_storage` helpers already present in `tests/test_loop.py` — confirm their exact existing signatures before running and mirror them (the existing file already constructs `Harness` and `build_registry`, so copy that call shape).
```
