"""
ThreatPipe v2 — Local quota accounting (no round-robin).

Tracks per provider:
  requests_today, requests_this_minute, input/output/estimated tokens,
  last_request_time, consecutive_failures, average_latency,
  estimated_remaining_quota.

429/quota -> cooldown + priority demotion.
5xx/timeout -> limited exponential backoff then failover (no endless retry).
"""

from __future__ import annotations

import threading
import time
from typing import Dict

from .base import ProviderConfig, UsageStats


class QuotaTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats: Dict[str, UsageStats] = {}
        self._day_start: Dict[str, float] = {}

    def _ensure(self, name: str) -> UsageStats:
        if name not in self._stats:
            self._stats[name] = UsageStats()
            self._day_start[name] = time.time()
        # daily reset (simple 24h window)
        if time.time() - self._day_start[name] > 24 * 3600:
            self._stats[name] = UsageStats()
            self._day_start[name] = time.time()
        return self._stats[name]

    def snapshot(self, name: str) -> UsageStats:
        with self._lock:
            s = self._ensure(name)
            # return a copy
            import copy
            return copy.copy(s)

    def all_snapshots(self) -> Dict[str, UsageStats]:
        with self._lock:
            return {k: self._snapshot_locked(k) for k in list(self._stats.keys())}

    def _snapshot_locked(self, name: str) -> UsageStats:
        import copy
        return copy.copy(self._ensure(name))

    def in_cooldown(self, name: str) -> bool:
        with self._lock:
            s = self._ensure(name)
            return time.time() < s.cooldown_until

    def check_quota(self, cfg: ProviderConfig, est_tokens: int = 1500) -> bool:
        """
        Conservative pre-flight check: minute + day RPM/TPM/RPD/TPD.
        Local providers (cost==0 and provider in ollama/local) bypass day caps
        since they have no cloud quota — still respect per-minute sanity.
        """
        with self._lock:
            s = self._ensure(cfg.provider_name)
            now = time.time()
            if now < s.cooldown_until:
                return False
            # minute window roll
            if now - s.window_start_minute >= 60:
                s.window_start_minute = now
                s.requests_this_minute = 0
            is_local = cfg.provider_name.lower() in ("ollama", "local", "llamacpp")
            if s.requests_this_minute + 1 > cfg.requests_per_minute:
                return False
            if not is_local:
                if s.requests_today + 1 > cfg.requests_per_day:
                    return False
                if s.estimated_tokens_today + est_tokens > cfg.tokens_per_day:
                    return False
                # tokens per minute approximated from estimated_tokens_today is coarse;
                # use minute requests as proxy + est_tokens guard
                if est_tokens > cfg.tokens_per_minute:
                    # single request bigger than TPM — still allow local, block cloud
                    return False
            return True

    def record_success(self, name: str, in_tokens: int, out_tokens: int, latency_ms: float):
        with self._lock:
            s = self._ensure(name)
            now = time.time()
            if now - s.window_start_minute >= 60:
                s.window_start_minute = now
                s.requests_this_minute = 0
            s.requests_this_minute += 1
            s.requests_today += 1
            s.total_requests += 1
            s.input_tokens_today += in_tokens
            s.output_tokens_today += out_tokens
            s.estimated_tokens_today += in_tokens + out_tokens
            s.last_request_time = now
            s.consecutive_failures = 0
            # running avg latency
            n = s.total_requests
            s.average_latency_ms = ((s.average_latency_ms * (n - 1)) + latency_ms) / max(n, 1)

    def record_failure(self, name: str, kind: str):
        with self._lock:
            s = self._ensure(name)
            s.consecutive_failures += 1
            s.total_failures += 1
            s.last_request_time = time.time()
            if kind in ("rate_limit", "quota_exceeded"):
                # 429 -> cooldown 60s * growth, cap 10 min. Do NOT endlessly retry.
                backoff = min(600, 60 * (2 ** min(s.consecutive_failures - 1, 3)))
                s.cooldown_until = time.time() + backoff
            elif kind in ("server_error", "timeout"):
                backoff = min(60, 2 ** min(s.consecutive_failures, 5))
                s.cooldown_until = time.time() + backoff

    def remaining_fraction(self, cfg: ProviderConfig) -> float:
        """0.0..1.0 estimated remaining daily quota (local == 1.0 always)."""
        with self._lock:
            if cfg.provider_name.lower() in ("ollama", "local", "llamacpp"):
                return 1.0
            s = self._ensure(cfg.provider_name)
            if time.time() < s.cooldown_until:
                return 0.0
            req_frac = max(0.0, 1.0 - s.requests_today / max(cfg.requests_per_day, 1))
            tok_frac = max(0.0, 1.0 - s.estimated_tokens_today / max(cfg.tokens_per_day, 1))
            base = min(req_frac, tok_frac)
            # penalize repeated failures
            penalty = 0.15 * min(s.consecutive_failures, 3)
            return max(0.0, base - penalty)
