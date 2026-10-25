"""ThreatPipe v2 — llm package public API.

App code should look like:
    from llm import generate
    resp = generate(messages, task_type=TaskType.BULK_PROCESSING)
while router decides provider/model/quota/fallback/retries/streaming.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import LLMRequest, LLMResponse, ProviderConfig, RoutingMode, TaskType
from .router import LLMRouter, get_router, reset_router

__all__ = [
    "TaskType", "RoutingMode", "LLMRequest", "LLMResponse", "ProviderConfig",
    "LLMRouter", "get_router", "reset_router",
    "generate", "chat", "stream",
]


def generate(messages: List[Dict[str, str]], task_type: TaskType = TaskType.NORMAL,
             **kw) -> LLMResponse:
    return get_router().generate(messages, task_type=task_type, **kw)


def chat(messages: List[Dict[str, str]], task_type: TaskType = TaskType.NORMAL,
         **kw) -> LLMResponse:
    return get_router().chat(messages, task_type=task_type, **kw)


def stream(messages: List[Dict[str, str]], task_type: TaskType = TaskType.NORMAL, **kw):
    return get_router().stream(messages, task_type=task_type, **kw)
