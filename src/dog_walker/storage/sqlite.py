from __future__ import annotations
import json
import sqlite3
from dog_walker.storage.base import Storage
from dog_walker.types import Message, RunRecord


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
