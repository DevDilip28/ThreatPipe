"""
ThreatPipe v2 — OpenAI-compatible provider adapter.

Serves: Jarvislabs Token Factory/Serverless, NVIDIA NIM, Groq,
local llama.cpp server, Ollama (/v1), OpenCode Zen gateway,
Gemini via OpenAI endpoint.
All are officially documented OpenAI-compatible /chat/completions APIs.
No quota bypass — standard auth + rate-limit handling only.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterator, List

import requests

from ..base import (
    LLMRequest, LLMResponse, ProviderAdapter, ProviderConfig,
    ProviderError, TaskType, classify_http_error,
)


class OpenAICompatProvider(ProviderAdapter):
    """Single adapter class parametrized by ProviderConfig (base_url/model/key)."""

    def _headers(self) -> Dict[str, str]:
        # Local providers need no auth
        if self.config.provider_name.lower() in ("ollama", "local", "llamacpp"):
            return {"Content-Type": "application/json"}
        key = os.getenv(self.config.api_key_env_var, "")
        # OpenCode Zen uses Bearer too
        return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    def _url(self, suffix: str = "/chat/completions") -> str:
        base = self.config.base_url.rstrip("/")
        # avoid double /v1/v1
        if base.endswith("/v1") and suffix.startswith("/"):
            return base + suffix
        if "opencode.ai/zen" in base and base.endswith("/zen/v1"):
            return base + "/chat/completions"
        return base.rstrip("/") + suffix

    def _payload(self, request: LLMRequest, model: str, stream: bool = False) -> Dict[str, Any]:
        send_model = model
        # Zen gateway expects bare id (e.g. "big-pickle", not "opencode/big-pickle")
        if self.name == "opencode" and send_model.startswith("opencode/"):
            send_model = send_model.split("/", 1)[1]
        p: Dict[str, Any] = {
            "model": send_model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream,
        }
        if request.tools and self.config.supports_tools:
            p["tools"] = request.tools
        return p

    def _parse(self, data: Dict[str, Any], model: str, latency_ms: float) -> LLMResponse:
        try:
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            text = msg.get("content") or choice.get("text") or ""
            # tool_calls passthrough as text JSON if present (for TOOL_CALLING tasks)
            if not text and msg.get("tool_calls"):
                text = json.dumps(msg.get("tool_calls"))
            usage = data.get("usage") or {}
            in_t = int(usage.get("prompt_tokens", 0))
            out_t = int(usage.get("completion_tokens", 0))
            tot = int(usage.get("total_tokens", in_t + out_t))
            if tot == 0:
                # rough estimate when provider omits usage
                tot = len(text) // 4
            return LLMResponse(text=text, provider=self.name, model=model,
                               input_tokens=in_t, output_tokens=out_t,
                               total_tokens=tot, latency_ms=latency_ms, raw=data)
        except Exception as exc:
            raise ProviderError(f"bad response: {exc}", kind="unknown")

    def chat(self, request: LLMRequest) -> LLMResponse:
        return self.generate(request)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.is_configured() and self.name.lower() not in ("ollama", "local", "llamacpp"):
            raise ProviderError(f"{self.name} not configured (missing {self.config.api_key_env_var})",
                                kind="auth")
        model = request.model_override or self.config.model
        t0 = time.monotonic()
        try:
            resp = requests.post(self._url(), headers=self._headers(),
                                 json=self._payload(request, model, stream=False),
                                 timeout=request.timeout_s or self.config.timeout_s)
        except requests.Timeout as exc:
            raise ProviderError(f"{self.name} timeout: {exc}", kind="timeout",
                                retryable=True)
        except Exception as exc:
            raise ProviderError(f"{self.name} transport: {exc}", kind="server_error",
                                retryable=True)
        latency = (time.monotonic() - t0) * 1000
        if resp.status_code != 200:
            raise classify_http_error(resp.status_code, resp.text)
        try:
            data = resp.json()
        except Exception:
            raise ProviderError("non-JSON response", kind="unknown")
        out = self._parse(data, model, latency)
        return out

    def stream(self, request: LLMRequest) -> Iterator[str]:
        if not self.config.supports_streaming:
            yield self.generate(request).text
            return
        model = request.model_override or self.config.model
        try:
            with requests.post(self._url(), headers=self._headers(),
                               json=self._payload(request, model, stream=True),
                               timeout=request.timeout_s or self.config.timeout_s,
                               stream=True) as resp:
                if resp.status_code != 200:
                    raise classify_http_error(resp.status_code, resp.text[:500])
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = ((chunk.get("choices") or [{}])[0].get("delta") or {})
                        piece = delta.get("content") or ""
                        if piece:
                            yield piece
                    except Exception:
                        continue
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"{self.name} stream: {exc}", kind="server_error", retryable=True)

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise ProviderError(f"{self.name} embeddings not wired (only if provider supports)",
                            kind="bad_request")

    def health_check(self) -> Dict[str, Any]:
        # local: try /models or /health; cloud: /models with auth (cheap, no inference)
        try:
            base = self.config.base_url.rstrip("/")
            url = base + "/models"
            h = self._headers()
            r = requests.get(url, headers=h, timeout=10)
            ok = r.status_code in (200, 401, 403, 404)
            # 401/403 means reachable but key wrong — report degraded, not down
            return {"provider": self.name, "ok": r.status_code == 200,
                    "status_code": r.status_code, "reachable": ok,
                    "model": self.config.model, "base_url": base}
        except Exception as exc:
            return {"provider": self.name, "ok": False, "error": str(exc)[:200]}

    def get_usage(self) -> Dict[str, Any]:
        # providers don't expose free-tier balances via API; router tracks locally
        return {"provider": self.name, "note": "tracked locally by QuotaTracker"}

    def get_model_info(self) -> Dict[str, Any]:
        from ..registry import get_model_capability
        cap = get_model_capability(self.config.model)
        return {"provider": self.name, "model": self.config.model,
                "base_url": self.config.base_url, "capability": cap,
                "supports_tools": self.config.supports_tools,
                "supports_streaming": self.config.supports_streaming,
                "context_window": self.config.context_window}
