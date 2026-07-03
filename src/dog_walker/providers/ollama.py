from __future__ import annotations
import ollama
from dog_walker.providers.base import Provider
from dog_walker.types import Message, ToolSpec, Response, ToolCall


def to_ollama_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "system":
            out.append({"role": "system", "content": m.text or ""})
        elif m.role == "user":
            out.append({"role": "user", "content": m.text or ""})
        elif m.role == "assistant":
            msg: dict = {"role": "assistant", "content": m.text or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.args}}
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            # Ollama expects one plain role="tool" message per result,
            # matched to the preceding call by order (no id needed).
            for r in m.tool_results:
                out.append({"role": "tool", "content": r.content})
    return out


def to_ollama_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": t.name, "description": t.description,
        "parameters": t.parameters}} for t in tools]


def parse_ollama_response(message) -> Response:
    text = getattr(message, "content", "") or None
    tool_calls: list[ToolCall] = []
    raw = getattr(message, "tool_calls", None) or []
    for i, tc in enumerate(raw):
        fn = tc.function
        # Ollama does not return a call id; synthesize a stable one by index.
        tool_calls.append(ToolCall(id=f"call_{i}", name=fn.name, args=dict(fn.arguments)))
    return Response(text=text, tool_calls=tool_calls)


class OllamaProvider(Provider):
    def __init__(self, model: str, host: str | None = None):
        self.model = model
        self.client = ollama.Client(host=host) if host else ollama.Client()

    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response:
        resp = self.client.chat(
            model=self.model,
            messages=to_ollama_messages(messages),
            tools=to_ollama_tools(tools),
            stream=False,
        )
        return parse_ollama_response(resp.message)
