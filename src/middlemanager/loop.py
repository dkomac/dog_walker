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
