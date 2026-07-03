from __future__ import annotations
from abc import ABC, abstractmethod
from dog_walker.types import Message, ToolSpec, Response


class Provider(ABC):
    @abstractmethod
    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response: ...
