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
