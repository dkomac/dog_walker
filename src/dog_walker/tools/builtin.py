from __future__ import annotations
import os
import subprocess
from dog_walker.tools.base import Tool, ToolRegistry


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


class SetPreference(Tool):
    name = "set_preference"
    description = (
        "Save a lasting user preference (e.g. 'be terse', 'use tabs'). Preferences are "
        "injected into your instructions on every future run. Use when the user states "
        "a standing preference about how you should behave."
    )
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "The preference to remember."}},
        "required": ["text"],
    }

    def __init__(self, preferences_file: str):
        self.preferences_file = preferences_file

    def run(self, text: str) -> str:
        with open(self.preferences_file, "a") as f:
            f.write(f"- {text}\n")
        return f"Saved preference: {text}"


def build_registry(enabled: list[str], confirm_bash: bool,
                   preferences_file: str = "preferences.md") -> ToolRegistry:
    available = {
        "read_file": ReadFile(),
        "write_file": WriteFile(),
        "list_files": ListFiles(),
        "run_bash": RunBash(confirm=confirm_bash),
        "set_preference": SetPreference(preferences_file),
    }
    reg = ToolRegistry()
    for name in enabled:
        if name in available:
            reg.register(available[name])
    return reg
