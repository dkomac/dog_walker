from __future__ import annotations
from middlemanager.providers.base import Provider
from middlemanager.types import Message, ToolSpec, Response


class FakeProvider(Provider):
    def __init__(self, scripted: list[Response]):
        self._scripted = list(scripted)
        self._i = 0
        self.calls: list[list[Message]] = []

    def send(self, messages: list[Message], tools: list[ToolSpec]) -> Response:
        self.calls.append(list(messages))
        resp = self._scripted[self._i]
        self._i += 1
        return resp
