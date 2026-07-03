from __future__ import annotations
from abc import ABC, abstractmethod
from dog_walker.types import ToolSpec


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
