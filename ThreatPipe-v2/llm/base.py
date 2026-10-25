"""
ThreatPipe v2 — LLM provider-independent abstraction (base types).

Application code must NOT import provider SDKs directly.
It talks only to these types + LLMRouter (router.py).

Interface required by spec:
    generate()
    stream()
    chat()
    embed()              # only if provider supports embeddings
    health_check()
    get_usage()
    get_model_info()
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class TaskType(str, Enum):
    SIMPLE = "SIMPLE"
    NORMAL = "NORMAL"
    CODING = "CODING"
    COMPLEX_REASONING = "COMPLEX_REASONING"
    LONG_CONTEXT = "LONG_CONTEXT"
    BULK_PROCESSING = "BULK_PROCESSING"
    TOOL_CALLING = "TOOL_CALLING"


class RoutingMode(str, Enum):
    QUALITY_FIRST = "QUALITY_FIRST"
    FREE_QUOTA_FIRST = "FREE_QUOTA_FIRST"
    HYBRID = "HYBRID"


@dataclass
class ProviderConfig:
    """Configurable provider descriptor. Limits are configurable, never hardcoded."""

    provider_name: str
    base_url: str
    model: str
    api_key_env_var: str  # e.g. GROQ_API_KEY — never store the key itself here
    priority: int = 100  # lower = tried earlier within same task chain
    capabilities: List[str] = field(default_factory=list)
    requests_per_minute: int = 30
    requests_per_day: int = 1000
    tokens_per_minute: int = 30000
    tokens_per_day: int = 500000
    cost: float = 0.0  # estimated USD per 1K tokens, 0.0 = free tier
    context_window: int = 8192
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_embeddings: bool = False
    enabled: bool = True
    extra_models: List[str] = field(default_factory=list)  # fallback models on same provider
    timeout_s: int = 60


@dataclass
class LLMRequest:
    messages: List[Dict[str, str]]  # [{"role": "system"|"user"|"assistant", "content": "..."}]
    task_type: TaskType = TaskType.NORMAL
    model_override: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2000
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None  # OpenAI-style tools, if needed
    timeout_s: int = 60


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    fallback_used: bool = False
    retry_count: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    requests_today: int = 0
    requests_this_minute: int = 0
    input_tokens_today: int = 0
    output_tokens_today: int = 0
    estimated_tokens_today: int = 0
    last_request_time: float = 0.0
    window_start_minute: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    average_latency_ms: float = 0.0
    estimated_remaining_quota: Optional[float] = None  # 0.0..1.0 fraction
    cooldown_until: float = 0.0  # epoch seconds; set on 429/quota
    total_requests: int = 0
    total_failures: int = 0


class ProviderError(Exception):
    """Base provider error with machine-readable kind."""

    def __init__(self, message: str, kind: str = "unknown", status_code: Optional[int] = None,
                 retryable: bool = False):
        super().__init__(message)
        self.kind = kind  # rate_limit | quota_exceeded | timeout | server_error | auth | bad_request | unknown
        self.status_code = status_code
        self.retryable = retryable


def classify_http_error(status_code: int, body_text: str = "") -> ProviderError:
    """Map HTTP status + body to ProviderError without leaking secrets."""
    low = (body_text or "").lower()
    if status_code == 429 or "rate limit" in low or "quota exceeded" in low or "insufficient quota" in low:
        return ProviderError(f"429/quota: {body_text[:200]}", kind="rate_limit",
                             status_code=status_code, retryable=False)
    if status_code in (500, 502, 503, 504):
        return ProviderError(f"transient {status_code}: {body_text[:200]}", kind="server_error",
                             status_code=status_code, retryable=True)
    if status_code in (408,) or "timeout" in low:
        return ProviderError(f"timeout: {body_text[:200]}", kind="timeout",
                             status_code=status_code, retryable=True)
    if status_code in (401, 403):
        return ProviderError("auth failed (check API key env var)", kind="auth",
                             status_code=status_code, retryable=False)
    return ProviderError(f"provider error {status_code}: {body_text[:200]}", kind="unknown",
                         status_code=status_code, retryable=False)


class ProviderAdapter(ABC):
    """Common interface every provider must implement."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def name(self) -> str:
        return self.config.provider_name

    def is_configured(self) -> bool:
        """Only env-var presence + enabled flag. No network call."""
        import os
        if not self.config.enabled:
            return False
        # local providers need no key
        if self.config.provider_name.lower() in ("ollama", "local", "llamacpp"):
            return True
        key = os.getenv(self.config.api_key_env_var, "")
        return bool(key and key.strip())

    # -- core ops --
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def chat(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    def stream(self, request: LLMRequest) -> Iterator[str]:
        # default: non-streaming fallback
        resp = self.chat(request)
        yield resp.text

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise ProviderError(f"{self.name} does not support embeddings", kind="bad_request")

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_usage(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        raise NotImplementedError
