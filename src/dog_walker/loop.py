from __future__ import annotations
import sys
from dog_walker.providers.base import Provider
from dog_walker.storage.base import Storage
from dog_walker.tools.base import ToolRegistry
from dog_walker.types import Message, ToolResult


def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in args.items())


def _truncate(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[:limit] + "…"


DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous assistant running inside a command-line harness. "
    "You have access to tools. When you need to read files, list directories, or run "
    "shell commands, you MUST invoke the provided tools using the tool-calling "
    "mechanism. Do NOT write a tool call as plain text or a code block, and never "
    "invent or imagine tool output. Once the tools have given you what you need, reply "
    "with a concise final answer in plain text and stop."
)


def build_system_prompt(cwd: str, tool_names: list[str], preferences: str = "") -> str:
    """Ground the model with its working directory, tools, and user preferences."""
    tools = ", ".join(tool_names) if tool_names else "(none)"
    prompt = (
        "You are an autonomous assistant running inside a command-line harness.\n"
        f"Your current working directory is: {cwd}\n"
        f"You have exactly these tools: {tools}. There are no other tools; do not "
        "invent tool names (there is no 'ls' tool — use 'list_files' or 'run_bash').\n"
        "To act, you MUST invoke a tool through the tool-calling mechanism. NEVER write "
        "a tool call as plain text or JSON in your reply, and never imagine tool output.\n"
        'To refer to the current directory, use "." — do NOT guess absolute paths '
        "like /home/user/project.\n"
        "When you have what you need, reply with a concise final answer in plain text "
        "and stop."
    )
    if preferences.strip():
        prompt += "\n\nUser preferences (always follow these):\n" + preferences.strip()
    return prompt


class Harness:
    def __init__(self, provider: Provider, registry: ToolRegistry,
                 storage: Storage, max_iterations: int,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 verbose: bool = False):
        self.provider = provider
        self.registry = registry
        self.storage = storage
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt
        self.verbose = verbose

    def _trace(self, msg: str) -> None:
        # Traces go to stderr so the final answer stays clean on stdout.
        if self.verbose:
            print(msg, file=sys.stderr)

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

        for step in range(1, self.max_iterations + 1):
            self._trace(f"\n── iteration {step} ──")
            response = self.provider.send(messages, self.registry.specs())

            assistant_msg = Message(role="assistant", text=response.text,
                                    tool_calls=response.tool_calls)
            messages.append(assistant_msg)
            self.storage.save_message(session_id, assistant_msg)

            if not response.tool_calls:
                self._trace("💬 final answer (no tool calls) → stopping")
                return response.text or ""

            results = []
            for call in response.tool_calls:
                self._trace(f"→ tool call: {call.name}({_fmt_args(call.args)})")
                output = self.registry.run(call.name, call.args)
                self._trace(f"← result: {_truncate(output)}")
                results.append(ToolResult(tool_call_id=call.id, content=output))

            tool_msg = Message(role="tool", tool_results=results)
            messages.append(tool_msg)
            self.storage.save_message(session_id, tool_msg)

        return f"Stopped: reached max iterations ({self.max_iterations})."
