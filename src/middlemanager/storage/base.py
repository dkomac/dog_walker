from __future__ import annotations
from abc import ABC, abstractmethod
from middlemanager.types import Message


class Storage(ABC):
    @abstractmethod
    def create_session(self) -> str: ...

    @abstractmethod
    def save_message(self, session_id: str, message: Message) -> None: ...

    @abstractmethod
    def load_messages(self, session_id: str) -> list[Message]: ...
