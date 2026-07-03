from __future__ import annotations
from dog_walker.providers.base import Provider
from dog_walker.storage.base import Storage
from dog_walker.tools.base import ToolRegistry
from dog_walker.types import Message, ToolResult


DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous assistant running inside a command-line harness. "
    "You have access to tools. When you need to read files, list directories, or run "
    "shell commands, you MUST invoke the provided tools using the tool-calling "
    "mechanism. Do NOT write a tool call as plain text or a code block, and never "
    "invent or imagine tool output. Once the tools have given you what you need, reply "
    "with a concise final answer in plain text and stop."
)


class Harness:
    def __init__(self, provider: Provider, registry: ToolRegistry,
                 storage: Storage, max_iterations: int,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.provider = provider
        self.registry = registry
        self.storage = storage
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

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
