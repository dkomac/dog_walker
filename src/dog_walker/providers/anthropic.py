from __future__ import annotations
from anthropic import Anthropic
from dog_walker.providers.base import Provider
from dog_walker.types import Message, ToolSpec, Response, ToolCall, Usage


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


def extract_system(messages: list[Message]) -> str | None:
    # Anthropic takes the system prompt as a separate top-level param, not a message.
    parts = [m.text for m in messages if m.role == "system" and m.text]
    return "\n".join(parts) if parts else None


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"name": t.name, "description": t.description,
             "input_schema": t.parameters} for t in tools]


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


class AnthropicProvider(Provider):
    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY

    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=to_anthropic_messages(messages),
            tools=to_anthropic_tools(tools),
        )
        system = extract_system(messages)
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = Usage(
                input_tokens=getattr(resp.usage, "input_tokens", None),
                output_tokens=getattr(resp.usage, "output_tokens", None),
            )
        return parse_anthropic_response(resp.content, usage)
