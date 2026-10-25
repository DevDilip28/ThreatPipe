"""
ThreatPipe v2 — Cloudflare Workers AI adapter (optional).

Only enabled when CLOUDFLARE_API_KEY (+ account id) is configured.
Uses officially documented Workers AI REST API.
If not configured, adapter reports disabled and router skips it.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import requests

from ..base import (
    LLMRequest, LLMResponse, ProviderAdapter, ProviderError, classify_http_error,
)


class CloudflareProvider(ProviderAdapter):
    def _creds(self):
        key = os.getenv(self.config.api_key_env_var, "")
        acct = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
        base = os.getenv("CLOUDFLARE_BASE_URL",
                         "https://api.cloudflare.com/client/v4/accounts")
        return key, acct, base

    def is_configured(self) -> bool:
        if not self.config.enabled:
            return False
        key, acct, _ = self._creds()
        return bool(key and acct)

    def _url(self) -> str:
        _, acct, base = self._creds()
        model = self.config.model
        return f"{base.rstrip('/')}/{acct}/ai/run/{model}"

    def generate(self, request: LLMRequest) -> LLMResponse:
        key, _, _ = self._creds()
        if not self.is_configured():
            raise ProviderError("cloudflare not configured (need CLOUDFLARE_API_KEY + ACCOUNT_ID)",
                                kind="auth")
        # Workers AI chat format: {"messages": [...]}
        t0 = time.monotonic()
        try:
            r = requests.post(self._url(),
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json={"messages": request.messages,
                                    "max_tokens": request.max_tokens},
                              timeout=request.timeout_s)
        except Exception as exc:
            raise ProviderError(f"cloudflare transport: {exc}", kind="server_error", retryable=True)
        lat = (time.monotonic() - t0) * 1000
        if r.status_code != 200:
            raise classify_http_error(r.status_code, r.text)
        try:
            data = r.json()
            res = data.get("result") or {}
            text = res.get("response") or res.get("content") or ""
            if isinstance(text, list):
                text = " ".join(str(x) for x in text)
        except Exception as exc:
            raise ProviderError(f"cloudflare parse: {exc}", kind="unknown")
        toks = len(text) // 4
        return LLMResponse(text=str(text), provider=self.name, model=self.config.model,
                           input_tokens=0, output_tokens=toks, total_tokens=toks,
                           latency_ms=lat, raw=data)

    def chat(self, request: LLMRequest) -> LLMResponse:
        return self.generate(request)

    def health_check(self) -> Dict[str, Any]:
        ok = self.is_configured()
        return {"provider": self.name, "ok": ok, "reachable": ok,
                "model": self.config.model}

    def get_usage(self) -> Dict[str, Any]:
        return {"provider": self.name, "note": "tracked locally"}

    def get_model_info(self) -> Dict[str, Any]:
        from ..registry import get_model_capability
        return {"provider": self.name, "model": self.config.model,
                "capability": get_model_capability(self.config.model),
                "supports_tools": False, "supports_streaming": False,
                "context_window": self.config.context_window}
