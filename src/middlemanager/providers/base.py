from __future__ import annotations
from abc import ABC, abstractmethod
from middlemanager.types import Message, ToolSpec, Response


class Provider(ABC):
    @abstractmethod
    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response: ...
