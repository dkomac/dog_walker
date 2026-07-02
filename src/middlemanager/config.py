from __future__ import annotations
import tomllib
from dataclasses import dataclass


@dataclass
class Config:
    provider_name: str
    model: str
    max_tokens: int
    max_iterations: int
    confirm_bash: bool
    enabled_tools: list[str]
    storage_backend: str
    storage_path: str


def load_config(path: str) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Config(
        provider_name=data["provider"]["name"],
        model=data["provider"]["model"],
        max_tokens=data["provider"]["max_tokens"],
        max_iterations=data["harness"]["max_iterations"],
        confirm_bash=data["harness"]["confirm_bash"],
        enabled_tools=data["tools"]["enabled"],
        storage_backend=data["storage"]["backend"],
        storage_path=data["storage"]["path"],
    )
