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
