# Slim Agent Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal, well-structured LLM agent loop in Python (against Claude) that can read/write files, list directories, and run shell commands, with swappable provider and storage backends — as a learning exercise.

**Architecture:** A central config-driven loop depends only on three interfaces — Provider (LLM backend), Storage (persistence), and a Tool registry. Concrete implementations (Anthropic, SQLite, four built-in tools) plug in behind those interfaces. A shared `types.py` holds the neutral message/tool/response shapes everything speaks.

**Tech Stack:** Python 3.11+ (for stdlib `tomllib`), `anthropic` SDK, `pytest`, stdlib `sqlite3`.

## Global Constraints

- Python 3.11 or newer (relies on stdlib `tomllib`).
- `loop.py` must NEVER import `anthropic` or `sqlite3` — only the interfaces in `providers/base.py`, `storage/base.py`, `tools/base.py`, and shared `types.py`.
- The Anthropic API key comes from the `ANTHROPIC_API_KEY` environment variable, never from config or code.
- Default model: `claude-haiku-4-5-20251001`.
- Package name (correct spelling): `middlemanager`. Repo directory: `/Users/d/Desktop/git/middlemanger`.
- Every task ends in an independently runnable/testable deliverable. TDD: test first. Commit after each task.

---

## File Structure

```
middlemanger/
├── config.toml                       # settings (exists)
├── data.db                           # sqlite file (exists, empty)
├── pyproject.toml                    # deps + pytest config          [Task 1]
├── .gitignore                                                        [Task 1]
├── src/middlemanager/
│   ├── __init__.py                                                   [Task 1]
│   ├── types.py                      # neutral shapes                [Task 2]
│   ├── config.py                     # parse config.toml            [Task 3]
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py                   # Storage interface            [Task 4]
│   │   └── sqlite.py                 # SQLite impl                  [Task 4]
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                   # Tool + registry             [Task 5]
│   │   └── builtin.py                # 4 built-in tools            [Task 5]
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                   # Provider interface          [Task 6]
│   │   ├── fake.py                   # fake provider (testing)     [Task 6]
│   │   └── anthropic.py             # Anthropic adapter           [Task 7]
│   ├── loop.py                       # THE HARNESS                 [Task 8]
│   └── main.py                       # entry point / CLI           [Task 9]
└── tests/                                                           [each task]
```

**Dependency order:** types → config → storage → tools → provider(interface+fake) → anthropic adapter → loop → main. The loop (Task 8) can be fully tested with the fake provider (Task 6) before the real adapter (Task 7) is even trusted.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/middlemanager/__init__.py`
- Create: `src/middlemanager/storage/__init__.py`, `src/middlemanager/tools/__init__.py`, `src/middlemanager/providers/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable package `middlemanager` and a working `pytest` command.

- [ ] **Step 1: Initialize git**

```bash
cd /Users/d/Desktop/git/middlemanger
git init
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
.pytest_cache/
data.db
.env
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[project]
name = "middlemanager"
version = "0.1.0"
description = "A slim LLM agent harness for learning."
requires-python = ">=3.11"
dependencies = ["anthropic>=0.40"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 4: Create the empty package files**

Create each of these as an empty file:
- `src/middlemanager/__init__.py`
- `src/middlemanager/storage/__init__.py`
- `src/middlemanager/tools/__init__.py`
- `src/middlemanager/providers/__init__.py`
- `tests/__init__.py`

- [ ] **Step 5: Create and activate a virtualenv, install dev deps**

```bash
cd /Users/d/Desktop/git/middlemanger
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
Expected: installs `anthropic`, `pytest`, and the local package without error.

- [ ] **Step 6: Verify pytest runs (no tests yet)**

Run: `pytest`
Expected: exits cleanly with "no tests ran" (exit code 5) — confirms discovery is wired.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: scaffold middlemanager package"
```

---

### Task 2: Neutral types

**Files:**
- Create: `src/middlemanager/types.py`
- Test: `tests/test_types.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ToolCall(id: str, name: str, args: dict)`
  - `ToolResult(tool_call_id: str, content: str)`
  - `ToolSpec(name: str, description: str, parameters: dict)`
  - `Message(role: str, text: str | None = None, tool_calls: list[ToolCall] = [], tool_results: list[ToolResult] = [])` with `to_dict()` and classmethod `from_dict(d)` for JSON persistence.
  - `Response(text: str | None, tool_calls: list[ToolCall])`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_types.py
from middlemanager.types import Message, ToolCall, ToolResult


def test_message_roundtrips_through_dict():
    msg = Message(
        role="assistant",
        text="let me look",
        tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a.txt"})],
    )
    restored = Message.from_dict(msg.to_dict())
    assert restored == msg


def test_tool_message_roundtrips():
    msg = Message(role="tool", tool_results=[ToolResult(tool_call_id="t1", content="hello")])
    assert Message.from_dict(msg.to_dict()) == msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'middlemanager.types'`

- [ ] **Step 3: Write the implementation**

```python
# src/middlemanager/types.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    tool_call_id: str
    content: str


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        return cls(
            role=d["role"],
            text=d.get("text"),
            tool_calls=[ToolCall(**c) for c in d.get("tool_calls", [])],
            tool_results=[ToolResult(**r) for r in d.get("tool_results", [])],
        )


@dataclass
class Response:
    text: str | None
    tool_calls: list[ToolCall]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/middlemanager/types.py tests/test_types.py
git commit -m "feat: add neutral message/tool/response types"
```

---

### Task 3: Config loading

**Files:**
- Create: `src/middlemanager/config.py`
- Modify: `config.toml` (currently empty — fill it in)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Config` dataclass with: `provider_name: str`, `model: str`, `max_tokens: int`, `max_iterations: int`, `confirm_bash: bool`, `enabled_tools: list[str]`, `storage_backend: str`, `storage_path: str`.
  - `load_config(path: str) -> Config`.

- [ ] **Step 1: Fill in `config.toml`**

```toml
[provider]
name = "anthropic"
model = "claude-haiku-4-5-20251001"
max_tokens = 4096

[harness]
max_iterations = 20
confirm_bash = true

[tools]
enabled = ["read_file", "write_file", "list_files", "run_bash"]

[storage]
backend = "sqlite"
path = "data.db"
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
from middlemanager.config import load_config


def test_load_config_reads_all_sections(tmp_path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        '[provider]\n'
        'name = "anthropic"\n'
        'model = "claude-haiku-4-5-20251001"\n'
        'max_tokens = 4096\n'
        '[harness]\n'
        'max_iterations = 20\n'
        'confirm_bash = true\n'
        '[tools]\n'
        'enabled = ["read_file", "run_bash"]\n'
        '[storage]\n'
        'backend = "sqlite"\n'
        'path = "data.db"\n'
    )
    cfg = load_config(str(cfg_file))
    assert cfg.provider_name == "anthropic"
    assert cfg.model == "claude-haiku-4-5-20251001"
    assert cfg.max_tokens == 4096
    assert cfg.max_iterations == 20
    assert cfg.confirm_bash is True
    assert cfg.enabled_tools == ["read_file", "run_bash"]
    assert cfg.storage_backend == "sqlite"
    assert cfg.storage_path == "data.db"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'middlemanager.config'`

- [ ] **Step 4: Write the implementation**

```python
# src/middlemanager/config.py
from __future__ import annotations
import tomllib
from dataclasses import dataclass


@dataclass
class Config:
    provider_name: str
    model: str
    max_tokens: int
    max_iterations: int
    confirm_bash: bool
    enabled_tools: list[str]
    storage_backend: str
    storage_path: str


def load_config(path: str) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Config(
        provider_name=data["provider"]["name"],
        model=data["provider"]["model"],
        max_tokens=data["provider"]["max_tokens"],
        max_iterations=data["harness"]["max_iterations"],
        confirm_bash=data["harness"]["confirm_bash"],
        enabled_tools=data["tools"]["enabled"],
        storage_backend=data["storage"]["backend"],
        storage_path=data["storage"]["path"],
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add config.toml src/middlemanager/config.py tests/test_config.py
git commit -m "feat: add config.toml loading"
```

---

### Task 4: Storage interface + SQLite implementation

**Files:**
- Create: `src/middlemanager/storage/base.py`
- Create: `src/middlemanager/storage/sqlite.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `Message` from `types.py`.
- Produces:
  - `Storage` (ABC) with: `create_session() -> str`, `save_message(session_id: str, message: Message) -> None`, `load_messages(session_id: str) -> list[Message]`.
  - `SqliteStorage(path: str)` implementing `Storage`. Creates tables on init. Session id is a monotonically increasing integer rendered as a string (avoids needing uuid/random). Messages preserve insertion order via an autoincrement `seq`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
from middlemanager.storage.sqlite import SqliteStorage
from middlemanager.types import Message, ToolCall


def test_save_and_load_roundtrip(tmp_path):
    db = str(tmp_path / "t.db")
    store = SqliteStorage(db)
    sid = store.create_session()

    store.save_message(sid, Message(role="user", text="hi"))
    store.save_message(
        sid,
        Message(role="assistant", text="ok",
                tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})]),
    )

    loaded = store.load_messages(sid)
    assert len(loaded) == 2
    assert loaded[0] == Message(role="user", text="hi")
    assert loaded[1].tool_calls[0].name == "read_file"


def test_sessions_are_isolated(tmp_path):
    store = SqliteStorage(str(tmp_path / "t.db"))
    s1 = store.create_session()
    s2 = store.create_session()
    store.save_message(s1, Message(role="user", text="in s1"))
    assert store.load_messages(s2) == []
    assert s1 != s2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the interface**

```python
# src/middlemanager/storage/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from middlemanager.types import Message


class Storage(ABC):
    @abstractmethod
    def create_session(self) -> str: ...

    @abstractmethod
    def save_message(self, session_id: str, message: Message) -> None: ...

    @abstractmethod
    def load_messages(self, session_id: str) -> list[Message]: ...
```

- [ ] **Step 4: Write the SQLite implementation**

```python
# src/middlemanager/storage/sqlite.py
from __future__ import annotations
import json
import sqlite3
from middlemanager.storage.base import Storage
from middlemanager.types import Message


class SqliteStorage(Storage):
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT NOT NULL,"
            " role TEXT NOT NULL,"
            " content TEXT NOT NULL,"
            " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )
        self.conn.commit()

    def create_session(self) -> str:
        cur = self.conn.execute("INSERT INTO sessions DEFAULT VALUES")
        self.conn.commit()
        return str(cur.lastrowid)

    def save_message(self, session_id: str, message: Message) -> None:
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, message.role, json.dumps(message.to_dict())),
        )
        self.conn.commit()

    def load_messages(self, session_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [Message.from_dict(json.loads(r[0])) for r in rows]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/middlemanager/storage/ tests/test_storage.py
git commit -m "feat: add Storage interface and SQLite backend"
```

---

### Task 5: Tool interface, registry, and built-in tools

**Files:**
- Create: `src/middlemanager/tools/base.py`
- Create: `src/middlemanager/tools/builtin.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `ToolSpec` from `types.py`.
- Produces:
  - `Tool` (ABC) with class attrs `name: str`, `description: str`, `parameters: dict`, and `run(self, **kwargs) -> str`.
  - `ToolRegistry` with `register(tool: Tool)`, `specs() -> list[ToolSpec]`, `run(name: str, args: dict) -> str`. `run` catches exceptions and returns `f"Error: {e}"` (errors become data fed back to the model).
  - Four tools: `ReadFile`, `WriteFile`, `ListFiles`, `RunBash`. `RunBash.__init__(self, confirm: bool)`; when `confirm` is true it prints the command and reads y/n from stdin, returning `"Command cancelled by user."` on anything but `y`.
  - `build_registry(enabled: list[str], confirm_bash: bool) -> ToolRegistry` that registers only the enabled tools by name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
from middlemanager.tools.base import ToolRegistry, Tool
from middlemanager.tools.builtin import ReadFile, WriteFile, ListFiles, build_registry


def test_read_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    assert ReadFile().run(path=str(f)) == "hello"


def test_write_then_read(tmp_path):
    f = str(tmp_path / "out.txt")
    WriteFile().run(path=f, content="data")
    assert ReadFile().run(path=f) == "data"


def test_list_files(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    (tmp_path / "y.txt").write_text("2")
    out = ListFiles().run(path=str(tmp_path))
    assert "x.txt" in out and "y.txt" in out


def test_registry_runs_by_name(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hi")
    reg = build_registry(["read_file"], confirm_bash=False)
    assert reg.run("read_file", {"path": str(f)}) == "hi"


def test_registry_turns_errors_into_data():
    reg = build_registry(["read_file"], confirm_bash=False)
    result = reg.run("read_file", {"path": "/nope/missing.txt"})
    assert result.startswith("Error:")


def test_specs_expose_enabled_tools_only():
    reg = build_registry(["read_file", "write_file"], confirm_bash=False)
    names = {s.name for s in reg.specs()}
    assert names == {"read_file", "write_file"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the interface + registry**

```python
# src/middlemanager/tools/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from middlemanager.types import ToolSpec


class Tool(ABC):
    name: str
    description: str
    parameters: dict  # JSON schema

    @abstractmethod
    def run(self, **kwargs) -> str: ...

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.parameters)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def run(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool {name!r}"
        try:
            return self._tools[name].run(**args)
        except Exception as e:  # errors become data for the model
            return f"Error: {e}"
```

- [ ] **Step 4: Write the built-in tools**

```python
# src/middlemanager/tools/builtin.py
from __future__ import annotations
import os
import subprocess
from middlemanager.tools.base import Tool, ToolRegistry


class ReadFile(Tool):
    name = "read_file"
    description = "Read and return the full text contents of a file at the given path."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the file."}},
        "required": ["path"],
    }

    def run(self, path: str) -> str:
        with open(path, "r") as f:
            return f.read()


class WriteFile(Tool):
    name = "write_file"
    description = "Write text content to a file at the given path, overwriting if it exists."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "content": {"type": "string", "description": "Text to write."},
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"


class ListFiles(Tool):
    name = "list_files"
    description = "List the names of files and directories at the given path."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path."}},
        "required": ["path"],
    }

    def run(self, path: str) -> str:
        return "\n".join(sorted(os.listdir(path)))


class RunBash(Tool):
    name = "run_bash"
    description = "Run a shell command and return its combined stdout and stderr."
    parameters = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Shell command."}},
        "required": ["command"],
    }

    def __init__(self, confirm: bool = True):
        self.confirm = confirm

    def run(self, command: str) -> str:
        if self.confirm:
            print(f"\n[run_bash] about to run: {command}")
            answer = input("Allow? (y/n): ").strip().lower()
            if answer != "y":
                return "Command cancelled by user."
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True
        )
        return (result.stdout or "") + (result.stderr or "")


def build_registry(enabled: list[str], confirm_bash: bool) -> ToolRegistry:
    available = {
        "read_file": ReadFile(),
        "write_file": WriteFile(),
        "list_files": ListFiles(),
        "run_bash": RunBash(confirm=confirm_bash),
    }
    reg = ToolRegistry()
    for name in enabled:
        if name in available:
            reg.register(available[name])
    return reg
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add src/middlemanager/tools/ tests/test_tools.py
git commit -m "feat: add tool registry and built-in tools"
```

---

### Task 6: Provider interface + fake provider

**Files:**
- Create: `src/middlemanager/providers/base.py`
- Create: `src/middlemanager/providers/fake.py`
- Test: `tests/test_fake_provider.py`

**Interfaces:**
- Consumes: `Message`, `ToolSpec`, `Response`, `ToolCall` from `types.py`.
- Produces:
  - `Provider` (ABC) with `send(self, messages: list[Message], tools: list[ToolSpec]) -> Response`.
  - `FakeProvider(scripted: list[Response])` implementing `Provider`. Each `send` returns the next scripted `Response`; records every `messages` argument it received in `self.calls: list[list[Message]]` for assertions.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fake_provider.py
from middlemanager.providers.fake import FakeProvider
from middlemanager.types import Response, ToolCall, Message


def test_fake_returns_scripted_responses_in_order():
    p = FakeProvider([
        Response(text=None, tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})]),
        Response(text="done", tool_calls=[]),
    ])
    r1 = p.send([Message(role="user", text="hi")], [])
    r2 = p.send([Message(role="user", text="hi")], [])
    assert r1.tool_calls[0].name == "read_file"
    assert r2.text == "done"


def test_fake_records_calls():
    p = FakeProvider([Response(text="ok", tool_calls=[])])
    p.send([Message(role="user", text="hi")], [])
    assert p.calls[0][0].text == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fake_provider.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the interface**

```python
# src/middlemanager/providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from middlemanager.types import Message, ToolSpec, Response


class Provider(ABC):
    @abstractmethod
    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response: ...
```

- [ ] **Step 4: Write the fake provider**

```python
# src/middlemanager/providers/fake.py
from __future__ import annotations
from middlemanager.providers.base import Provider
from middlemanager.types import Message, ToolSpec, Response


class FakeProvider(Provider):
    def __init__(self, scripted: list[Response]):
        self._scripted = list(scripted)
        self._i = 0
        self.calls: list[list[Message]] = []

    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response:
        self.calls.append(list(messages))
        resp = self._scripted[self._i]
        self._i += 1
        return resp
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_fake_provider.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add src/middlemanager/providers/base.py src/middlemanager/providers/fake.py tests/test_fake_provider.py
git commit -m "feat: add Provider interface and fake provider for testing"
```

---

### Task 7: Anthropic adapter

**Files:**
- Create: `src/middlemanager/providers/anthropic.py`
- Test: `tests/test_anthropic_adapter.py` (translation only — no live API)

**Interfaces:**
- Consumes: `Provider` from `providers/base.py`; `Message`, `ToolSpec`, `Response`, `ToolCall`, `ToolResult` from `types.py`; the `anthropic` SDK.
- Produces:
  - `AnthropicProvider(model: str, max_tokens: int)` implementing `Provider`. Reads the key from `ANTHROPIC_API_KEY` via the SDK's default behavior.
  - Two pure module-level translation functions, testable without the network:
    - `to_anthropic_messages(messages: list[Message]) -> list[dict]`
    - `parse_anthropic_response(content_blocks: list) -> Response`

**Translation rules (the heart of the adapter):**
- `Message(role="user", text=t)` → `{"role": "user", "content": t}`
- `Message(role="assistant", text=t, tool_calls=[...])` → `{"role": "assistant", "content": [ {"type":"text","text":t} (if t), {"type":"tool_use","id":c.id,"name":c.name,"input":c.args} for each call ]}`
- `Message(role="tool", tool_results=[...])` → `{"role": "user", "content": [ {"type":"tool_result","tool_use_id":r.tool_call_id,"content":r.content} for each result ]}`
- Response parse: iterate content blocks; collect `text` blocks into `text`; collect `tool_use` blocks into `ToolCall(id, name, args=input)`.

- [ ] **Step 1: Write the failing tests (translation logic)**

```python
# tests/test_anthropic_adapter.py
from types import SimpleNamespace
from middlemanager.providers.anthropic import to_anthropic_messages, parse_anthropic_response
from middlemanager.types import Message, ToolCall, ToolResult


def test_user_message_translation():
    out = to_anthropic_messages([Message(role="user", text="hi")])
    assert out == [{"role": "user", "content": "hi"}]


def test_assistant_with_tool_call_translation():
    msg = Message(role="assistant", text="looking",
                  tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "a"})])
    out = to_anthropic_messages([msg])[0]
    assert out["role"] == "assistant"
    assert {"type": "text", "text": "looking"} in out["content"]
    assert {"type": "tool_use", "id": "t1", "name": "read_file",
            "input": {"path": "a"}} in out["content"]


def test_tool_result_translation():
    msg = Message(role="tool", tool_results=[ToolResult(tool_call_id="t1", content="hello")])
    out = to_anthropic_messages([msg])[0]
    assert out == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "hello"}]}


def test_parse_response_with_text_and_tool_use():
    blocks = [
        SimpleNamespace(type="text", text="let me check"),
        SimpleNamespace(type="tool_use", id="t9", name="run_bash", input={"command": "ls"}),
    ]
    resp = parse_anthropic_response(blocks)
    assert resp.text == "let me check"
    assert resp.tool_calls == [ToolCall(id="t9", name="run_bash", args={"command": "ls"})]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_anthropic_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the adapter**

```python
# src/middlemanager/providers/anthropic.py
from __future__ import annotations
from anthropic import Anthropic
from middlemanager.providers.base import Provider
from middlemanager.types import Message, ToolSpec, Response, ToolCall


def to_anthropic_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text or ""})
        elif m.role == "assistant":
            content: list[dict] = []
            if m.text:
                content.append({"type": "text", "text": m.text})
            for c in m.tool_calls:
                content.append({"type": "tool_use", "id": c.id,
                                "name": c.name, "input": c.args})
            out.append({"role": "assistant", "content": content})
        elif m.role == "tool":
            content = [{"type": "tool_result", "tool_use_id": r.tool_call_id,
                        "content": r.content} for r in m.tool_results]
            out.append({"role": "user", "content": content})
    return out


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description,
             "input_schema": t.parameters} for t in tools]


def parse_anthropic_response(content_blocks: list) -> Response:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))
    text = "\n".join(text_parts) if text_parts else None
    return Response(text=text, tool_calls=tool_calls)


class AnthropicProvider(Provider):
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY

    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=to_anthropic_messages(messages),
            tools=to_anthropic_tools(tools),
        )
        return parse_anthropic_response(resp.content)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_anthropic_adapter.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/middlemanager/providers/anthropic.py tests/test_anthropic_adapter.py
git commit -m "feat: add Anthropic provider adapter"
```

---

### Task 8: The agent loop

**Files:**
- Create: `src/middlemanager/loop.py`
- Test: `tests/test_loop.py`

**Interfaces:**
- Consumes: `Provider` (base), `Storage` (base), `ToolRegistry`, and `Message`, `Response`, `ToolCall`, `ToolResult` from `types.py`.
- Produces:
  - `Harness(provider, registry, storage, max_iterations)` with `run(user_input: str) -> str`.
  - Loop behavior: create a session; append+save the user message; each iteration call `provider.send(messages, registry.specs())`, save+append the assistant message; if no tool calls → return the text; else run each tool call, build one `Message(role="tool", tool_results=[...])`, save+append it, loop. On hitting `max_iterations`, return `"Stopped: reached max iterations (N)."`.

- [ ] **Step 1: Write the failing tests (using the fake provider)**

```python
# tests/test_loop.py
from middlemanager.loop import Harness
from middlemanager.providers.fake import FakeProvider
from middlemanager.storage.sqlite import SqliteStorage
from middlemanager.tools.builtin import build_registry
from middlemanager.types import Response, ToolCall


def _storage(tmp_path):
    return SqliteStorage(str(tmp_path / "t.db"))


def test_returns_text_when_no_tool_calls(tmp_path):
    provider = FakeProvider([Response(text="hello there", tool_calls=[])])
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5)
    assert harness.run("hi") == "hello there"


def test_runs_tool_then_finishes(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("file contents")
    provider = FakeProvider([
        Response(text=None, tool_calls=[ToolCall(id="t1", name="read_file",
                                                 args={"path": str(f)})]),
        Response(text="the file says: file contents", tool_calls=[]),
    ])
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=5)
    result = harness.run("read the file")
    assert result == "the file says: file contents"
    # Second call to the provider must include the tool result in the messages.
    second_call_messages = provider.calls[1]
    assert any(m.role == "tool" and m.tool_results[0].content == "file contents"
               for m in second_call_messages)


def test_stops_at_max_iterations(tmp_path):
    # Provider always asks for another tool call -> never terminates on its own.
    looping = [Response(text=None, tool_calls=[ToolCall(id=f"t{i}", name="read_file",
                                                        args={"path": "/nope"})])
               for i in range(10)]
    provider = FakeProvider(looping)
    reg = build_registry(["read_file"], confirm_bash=False)
    harness = Harness(provider, reg, _storage(tmp_path), max_iterations=3)
    result = harness.run("go")
    assert "max iterations" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the loop**

```python
# src/middlemanager/loop.py
from __future__ import annotations
from middlemanager.providers.base import Provider
from middlemanager.storage.base import Storage
from middlemanager.tools.base import ToolRegistry
from middlemanager.types import Message, ToolResult


class Harness:
    def __init__(self, provider: Provider, registry: ToolRegistry,
                 storage: Storage, max_iterations: int):
        self.provider = provider
        self.registry = registry
        self.storage = storage
        self.max_iterations = max_iterations

    def run(self, user_input: str) -> str:
        session_id = self.storage.create_session()
        messages: list[Message] = []

        user_msg = Message(role="user", text=user_input)
        messages.append(user_msg)
        self.storage.save_message(session_id, user_msg)

        for _ in range(self.max_iterations):
            response = self.provider.send(messages, self.registry.specs())

            assistant_msg = Message(role="assistant", text=response.text,
                                    tool_calls=response.tool_calls)
            messages.append(assistant_msg)
            self.storage.save_message(session_id, assistant_msg)

            if not response.tool_calls:
                return response.text or ""

            results = []
            for call in response.tool_calls:
                output = self.registry.run(call.name, call.args)
                results.append(ToolResult(tool_call_id=call.id, content=output))

            tool_msg = Message(role="tool", tool_results=results)
            messages.append(tool_msg)
            self.storage.save_message(session_id, tool_msg)

        return f"Stopped: reached max iterations ({self.max_iterations})."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the whole suite**

Run: `pytest`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/middlemanager/loop.py tests/test_loop.py
git commit -m "feat: add the agent loop"
```

---

### Task 9: Wiring & CLI entry point

**Files:**
- Create: `src/middlemanager/main.py`
- Modify: `pyproject.toml` (add a console script entry point)
- Test: `tests/test_wiring.py`

**Interfaces:**
- Consumes: `load_config`, `SqliteStorage`, `build_registry`, `AnthropicProvider`, `FakeProvider`, `Harness`.
- Produces:
  - `build_provider(cfg) -> Provider` — returns `AnthropicProvider` when `cfg.provider_name == "anthropic"`, else raises `ValueError(f"Unknown provider: {name}")`.
  - `build_storage(cfg) -> Storage` — returns `SqliteStorage(cfg.storage_path)` when `cfg.storage_backend == "sqlite"`, else raises `ValueError`.
  - `build_harness(cfg) -> Harness` — wires provider + registry + storage.
  - `main()` — parse a single CLI arg (the prompt) or read from stdin, load `config.toml`, build the harness, run it, print the result. Wrap `provider.send` failures so an API error prints cleanly instead of a traceback (the loop itself stays simple; catch at `main`).

- [ ] **Step 1: Write the failing tests (factory dispatch — no network)**

```python
# tests/test_wiring.py
import pytest
from middlemanager.config import Config
from middlemanager.main import build_storage, build_provider
from middlemanager.storage.sqlite import SqliteStorage


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_wiring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write `main.py`**

```python
# src/middlemanager/main.py
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
```

Note: `build_provider` imports `AnthropicProvider` lazily so the wiring tests (and any
future non-Anthropic run) don't require the `anthropic` package or an API key to import
`main`.

- [ ] **Step 4: Add the console script to `pyproject.toml`**

Add this section:

```toml
[project.scripts]
middlemanager = "middlemanager.main:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_wiring.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Reinstall so the console script registers**

```bash
pip install -e ".[dev]"
```

- [ ] **Step 7: Full suite**

Run: `pytest`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/middlemanager/main.py pyproject.toml tests/test_wiring.py
git commit -m "feat: wire config-driven harness and CLI entry point"
```

---

### Task 10: End-to-end live smoke test (manual)

**Files:** none (manual verification).

**Interfaces:** Consumes the installed `middlemanager` console script and a live API key.

This is the payoff — the first real run against Claude. Not automated (costs money, needs a key).

- [ ] **Step 1: Set the API key**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

- [ ] **Step 2: A read-only task**

```bash
middlemanager "List the files in the current directory, then tell me what this project is."
```
Expected: the model calls `list_files`, possibly `read_file` on config.toml / the spec, then prints a summary. No bash confirmation prompt (it didn't need bash).

- [ ] **Step 3: A task that triggers the bash confirmation**

```bash
middlemanager "Show me the current git status."
```
Expected: a `[run_bash] about to run: git status` prompt appears; answer `y`; the model reports the status.

- [ ] **Step 4: Verify persistence**

```bash
sqlite3 data.db "SELECT role, substr(content,1,60) FROM messages ORDER BY id;"
```
Expected: rows for the user / assistant / tool messages from the runs above.

- [ ] **Step 5: Confirm the swap story (no code run — just verify by reading)**

Confirm `loop.py` imports only `providers.base`, `storage.base`, `tools.base`, and
`types` — never `anthropic` or `sqlite3`. This is the proof that adding an OpenAI/Ollama
provider or a Postgres backend later is a drop-in.

---

## Self-Review Notes

- **Spec coverage:** loop ✓ (T8), four tools ✓ (T5), config.toml ✓ (T3), SQLite persistence ✓ (T4), Provider abstraction ✓ (T6/T7), Storage abstraction ✓ (T4), fake-provider testing ✓ (T6/T8), bash confirmation safety ✓ (T5), errors-as-data ✓ (T5 registry), max_iterations cap ✓ (T8), config-driven backend selection ✓ (T9). Deferred items (multi-agent, OpenAI/Ollama, Postgres) intentionally absent.
- **Deviation from spec:** added `types.py` as a shared neutral-types module (spec implied these types lived in `providers/base.py`); this avoids circular imports since storage, tools, provider, and loop all reference them. Noted here and in the plan header.
- **Type consistency:** `Message`, `ToolCall`, `ToolResult`, `ToolSpec`, `Response` signatures are defined once in T2 and used verbatim in T4–T9. `build_registry(enabled, confirm_bash)`, `Harness(provider, registry, storage, max_iterations)`, `FakeProvider(scripted)` signatures match across their definitions and call sites.
- **v1 simplification:** the loop starts a fresh session each run (does not replay prior history), though `Storage.load_messages` is implemented and tested for later use. This matches the spec's "or start fresh."
