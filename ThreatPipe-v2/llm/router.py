"""
ThreatPipe v2 — Intelligent LLM router.

Flow per request:
  Application -> LLM Router -> select provider/model -> execute
  -> record usage -> return response

Considers: task type, capability, health, remaining quota,
token cost, rate, latency, context size, tool/streaming support.

No round-robin. No endless retry (max 1 retry per provider for
transient 5xx/timeout with backoff, then failover to next).
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from .base import (
    LLMRequest, LLMResponse, ProviderAdapter, ProviderConfig,
    ProviderError, RoutingMode, TaskType,
)
from .provider_config import (
    default_provider_configs, fallback_chains, force_provider, routing_mode,
)
from .providers.factory import build_adapters
from .quota import QuotaTracker
from .registry import get_model_capability


def _get_logger():
    try:
        from logger import get_logger
        return get_logger("llm_router")
    except Exception:
        import logging
        return logging.getLogger("llm_router")


TASK_POLICY = {
    # cheaper/faster for SIMPLE/BULK, stronger only when required
    TaskType.SIMPLE: {"prefer": ["speed", "reliability"], "min_reasoning": 0.0, "need_tools": False},
    TaskType.BULK_PROCESSING: {"prefer": ["speed", "reliability"], "min_reasoning": 0.0, "need_tools": False},
    TaskType.NORMAL: {"prefer": ["reliability", "reasoning_quality"], "min_reasoning": 0.5, "need_tools": False},
    TaskType.TOOL_CALLING: {"prefer": ["tool_use_quality", "reliability"], "min_reasoning": 0.5, "need_tools": True},
    TaskType.CODING: {"prefer": ["coding_quality", "reasoning_quality"], "min_reasoning": 0.6, "need_tools": False},
    TaskType.COMPLEX_REASONING: {"prefer": ["reasoning_quality", "reliability"], "min_reasoning": 0.65, "need_tools": False},
    TaskType.LONG_CONTEXT: {"prefer": ["context_length", "reliability"], "min_reasoning": 0.5, "need_tools": False},
}


class LLMRouter:
    def __init__(self, configs: Optional[List[ProviderConfig]] = None,
                 mode: Optional[str] = None):
        self.configs = configs or default_provider_configs()
        self.by_name = {c.provider_name: c for c in self.configs}
        self.adapters: Dict[str, ProviderAdapter] = build_adapters(self.configs)
        self.quota = QuotaTracker()
        self.mode = RoutingMode(mode or routing_mode())
        self.chains = fallback_chains()
        self.log = _get_logger()
        # structured request log (no keys, no prompt contents)
        self._history: List[Dict[str, Any]] = []

    # -- discovery --
    def available_providers(self) -> List[str]:
        return [n for n, a in self.adapters.items() if a.is_configured()]

    def _chain_for(self, task: TaskType) -> List[str]:
        forced = force_provider()
        if forced:
            if forced in self.adapters and self.adapters[forced].is_configured():
                return [forced]
            return []
        chain = list(self.chains.get(task.value, []) or self.chains.get(task.name, []))
        if not chain:
            chain = sorted(self.by_name.keys(), key=lambda n: self.by_name[n].priority)
        # keep only configured + enabled, preserve chain order
        return [n for n in chain if n in self.adapters and self.adapters[n].is_configured()]

    def _score(self, name: str, task: TaskType, est_tokens: int, need_stream: bool) -> float:
        cfg = self.by_name[name]
        cap = get_model_capability(cfg.model)
        pol = TASK_POLICY.get(task, TASK_POLICY[TaskType.NORMAL])

        if pol.get("need_tools") and not cfg.supports_tools:
            return -1.0
        if need_stream and not cfg.supports_streaming:
            return -1.0
        if task == TaskType.LONG_CONTEXT and cfg.context_window < est_tokens + 1000:
            return -1.0
        if cap.get("reasoning_quality", 0) < pol.get("min_reasoning", 0) - 0.25:
            # allow weaker local for BULK even if reasoning low
            if task not in (TaskType.SIMPLE, TaskType.BULK_PROCESSING):
                return -1.0

        remaining = self.quota.remaining_fraction(cfg)  # 0..1
        if remaining <= 0.01:
            return -1.0
        snap = self.quota.snapshot(name)
        # Mild tie-breaker only: normal 3-5s cloud latency must not outrank
        # a configured primary provider (that caused a 60s Gemini timeout).
        latency_penalty = min(0.2, max(0.0, (snap.average_latency_ms - 5000.0) / 100000.0))
        # Task-specific boosts (configurable via fallback_chains order + this):
        # BULK prefers local/Ollama (no quota, offline-friendly) per spec.
        bulk_boost = 0.30 if (task == TaskType.BULK_PROCESSING
                              and name.lower() in ("local", "ollama", "llamacpp")) else 0.0
        # Respect configured fallback chain order as tie-breaker signal
        try:
            chain = self._chain_for(task)
            pos = chain.index(name) if name in chain else len(chain)
            chain_boost = (len(chain) - pos) * 0.02
        except Exception:
            chain_boost = 0.0

        if self.mode == RoutingMode.QUALITY_FIRST:
            q = 0.0
            for k in pol["prefer"]:
                v = cap.get(k, 0.5)
                if k == "context_length":
                    v = min(1.0, float(v) / 131072.0)
                if k == "speed":
                    q += float(v) * 0.5
                else:
                    q += float(v)
            q = q / max(len(pol["prefer"]), 1)
            return q * 0.7 + remaining * 0.3 - latency_penalty + bulk_boost + chain_boost
        elif self.mode == RoutingMode.FREE_QUOTA_FIRST:
            # best usefulness per remaining quota: speed*reliability weighted by remaining
            usefulness = (cap.get("speed", 0.5) * 0.4 + cap.get("reliability", 0.5) * 0.3
                          + cap.get("reasoning_quality", 0.5) * 0.3)
            # local (no quota) naturally scores high via remaining==1.0
            return usefulness * 0.4 + remaining * 0.6 - latency_penalty + bulk_boost + chain_boost
        else:  # HYBRID
            q = sum(float(cap.get(k, 0.5)) if k not in ("context_length", "speed")
                    else (min(1.0, float(cap.get(k, 0.5)) / 131072.0) if k == "context_length"
                          else float(cap.get(k, 0.5)) * 0.5)
                    for k in pol["prefer"]) / max(len(pol["prefer"]), 1)
            return q * 0.5 + remaining * 0.5 - latency_penalty + bulk_boost + chain_boost

    def select(self, task: TaskType, est_tokens: int = 1500, need_stream: bool = False,
               need_tools: bool = False) -> List[str]:
        """Ordered candidate provider names."""
        cands = self._chain_for(task)
        if need_tools:
            # hard filter tools when explicitly required
            cands = [n for n in cands if self.by_name[n].supports_tools]
        scored = []
        for n in cands:
            if not self.quota.check_quota(self.by_name[n], est_tokens):
                continue
            s = self._score(n, task, est_tokens, need_stream)
            if s >= 0:
                scored.append((s, self.by_name[n].priority, n))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [n for _, _, n in scored]

    # -- execution --
    def _log_req(self, entry: Dict[str, Any]):
        # never log keys or prompt contents — only lengths + metadata
        try:
            self.log.info(
                f"LLM {entry.get('task_type')} provider={entry.get('provider')} "
                f"model={entry.get('model')} ok={entry.get('success')} "
                f"lat={entry.get('latency_ms', 0):.0f}ms in={entry.get('input_tokens')} "
                f"out={entry.get('output_tokens')} retry={entry.get('retry_count')} "
                f"fallback={entry.get('fallback_used')} err={entry.get('error_type')}"
            )
        except Exception:
            pass
        self._history.append(entry)
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def generate(self, messages: List[Dict[str, str]], task_type: TaskType = TaskType.NORMAL,
                 temperature: float = 0.0, max_tokens: int = 2000,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 model_override: Optional[str] = None,
                 timeout_s: int = 60) -> LLMResponse:
        est = sum(len(m.get("content", "")) for m in messages) // 4 + max_tokens
        cands = self.select(task_type, est_tokens=est, need_tools=bool(tools))
        if not cands:
            raise ProviderError("no configured providers with quota (check env keys + local server)",
                                kind="quota_exceeded")
        last_err: Optional[Exception] = None
        for idx, name in enumerate(cands):
            adapter = self.adapters[name]
            req = LLMRequest(messages=messages, task_type=task_type, temperature=temperature,
                             max_tokens=max_tokens, tools=tools,
                             model_override=model_override, timeout_s=timeout_s)
            # Throttle every provider call by 5s default to reduce rate-limit
            # blocking. Configurable via LLM_CALL_DELAY_S / llm_routing.call_delay_s.
            try:
                from .provider_config import call_delay_s as _delay
                _d = _delay()
            except Exception:
                _d = 5.0
            if _d > 0:
                try:
                    self.log.debug(f"throttle sleep {_d:.1f}s before {name}")
                except Exception:
                    pass
                time.sleep(_d)
            t0 = time.monotonic()
            try:
                resp = adapter.generate(req)
                lat = (time.monotonic() - t0) * 1000
                resp.latency_ms = lat
                resp.fallback_used = idx > 0
                resp.retry_count = idx
                self.quota.record_success(name, resp.input_tokens, resp.output_tokens, lat)
                self._log_req({"timestamp": datetime.now(timezone.utc).isoformat(),
                               "task_type": task_type.value, "provider": name,
                               "model": resp.model, "latency_ms": lat,
                               "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                               "total_tokens": resp.total_tokens, "success": True,
                               "retry_count": idx, "fallback_used": idx > 0, "error_type": None})
                return resp
            except ProviderError as exc:
                lat = (time.monotonic() - t0) * 1000
                # transient 5xx/timeout: ONE retry with backoff, then failover
                if exc.retryable and exc.kind in ("server_error", "timeout"):
                    time.sleep(min(4, 2 ** 1))
                    try:
                        t1 = time.monotonic()
                        resp2 = adapter.generate(req)
                        lat2 = (time.monotonic() - t1) * 1000
                        resp2.latency_ms = lat2
                        resp2.fallback_used = idx > 0
                        resp2.retry_count = idx + 1
                        self.quota.record_success(name, resp2.input_tokens, resp2.output_tokens, lat2)
                        self._log_req({"timestamp": datetime.now(timezone.utc).isoformat(),
                                       "task_type": task_type.value, "provider": name,
                                       "model": resp2.model, "latency_ms": lat2,
                                       "input_tokens": resp2.input_tokens,
                                       "output_tokens": resp2.output_tokens,
                                       "total_tokens": resp2.total_tokens, "success": True,
                                       "retry_count": idx + 1, "fallback_used": idx > 0,
                                       "error_type": None})
                        return resp2
                    except ProviderError as exc2:
                        exc = exc2
                self.quota.record_failure(name, exc.kind)
                self._log_req({"timestamp": datetime.now(timezone.utc).isoformat(),
                               "task_type": task_type.value, "provider": name,
                               "model": adapter.config.model, "latency_ms": lat,
                               "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                               "success": False, "retry_count": idx,
                               "fallback_used": idx > 0, "error_type": exc.kind})
                last_err = exc
                # auth errors: skip quickly; quota: cooldown already set, try next
                continue
        raise last_err or ProviderError("all providers failed", kind="unknown")

    # convenience wrappers so app code looks like llm.generate(...)
    def chat(self, *a, **k) -> LLMResponse:
        return self.generate(*a, **k)

    def stream(self, messages, task_type: TaskType = TaskType.NORMAL, **kw) -> Iterator[str]:
        est = sum(len(m.get("content", "")) for m in messages) // 4 + kw.get("max_tokens", 2000)
        cands = self.select(task_type, est_tokens=est, need_stream=True)
        if not cands:
            raise ProviderError("no streaming-capable provider available", kind="quota_exceeded")
        last = None
        for name in cands:
            try:
                req = LLMRequest(messages=messages, task_type=task_type,
                                 temperature=kw.get("temperature", 0.0),
                                 max_tokens=kw.get("max_tokens", 2000),
                                 timeout_s=kw.get("timeout_s", 60))
                yield from self.adapters[name].stream(req)
                return
            except Exception as exc:
                last = exc
                continue
        raise last or ProviderError("stream failed", kind="unknown")

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise ProviderError("embeddings not enabled (only if provider supports)", kind="bad_request")

    def health_check(self) -> Dict[str, Any]:
        return {n: a.health_check() for n, a in self.adapters.items()}

    def get_usage(self) -> Dict[str, Any]:
        out = {}
        for n, cfg in self.by_name.items():
            s = self.quota.snapshot(n)
            out[n] = {"model": cfg.model, "enabled": cfg.enabled,
                      "configured": self.adapters[n].is_configured(),
                      "requests_today": s.requests_today,
                      "estimated_tokens_today": s.estimated_tokens_today,
                      "remaining": round(self.quota.remaining_fraction(cfg), 3),
                      "avg_latency_ms": round(s.average_latency_ms, 1),
                      "failures": s.total_failures,
                      "cooldown": s.cooldown_until > time.time()}
        return out

    def get_model_info(self) -> Dict[str, Any]:
        return {n: a.get_model_info() for n, a in self.adapters.items()}

    def route_preview(self, task_type: TaskType, est_tokens: int = 1500) -> List[Dict[str, Any]]:
        order = self.select(task_type, est_tokens=est_tokens)
        return [{"provider": n, "model": self.by_name[n].model,
                 "remaining": round(self.quota.remaining_fraction(self.by_name[n]), 3),
                 "score": round(self._score(n, task_type, est_tokens, False), 3)}
                for n in order]


_singleton: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    global _singleton
    if _singleton is None:
        # .env support without committing secrets
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        _singleton = LLMRouter()
    return _singleton


def reset_router():
    global _singleton
    _singleton = None
