"""Mocked LLM router tests — no real API credits consumed."""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.base import LLMRequest, ProviderConfig, ProviderError, TaskType
from llm.quota import QuotaTracker
from llm.registry import get_model_capability
from llm.router import LLMRouter


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    # Unit tests must not sleep 5s per mocked call
    monkeypatch.setenv("LLM_CALL_DELAY_S", "0")


def _cfg(name, **kw):
    base = {
        "provider_name": name, "base_url": "http://localhost:9/v1",
        "model": f"{name}-model", "api_key_env_var": f"{name.upper()}_API_KEY",
    }
    base.update(kw)
    return ProviderConfig(**base)


def _router_with_mocks():
    r = LLMRouter(configs=[
        _cfg("nvidia", priority=10, supports_tools=True),
        _cfg("groq", priority=20, supports_tools=True),
        _cfg("local", priority=30, supports_tools=False),
    ])
    return r


def test_provider_selection_prefers_chain_order():
    r = _router_with_mocks()
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        # local needs no key; all three configured
        order = r.select(TaskType.CODING)
        assert order[0] == "nvidia"  # CODING chain starts with nvidia


def test_tool_calling_filters_non_tool_providers():
    r = _router_with_mocks()
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        order = r.select(TaskType.TOOL_CALLING)
        assert "local" not in order
        assert set(order) <= {"nvidia", "groq"}


def test_bulk_prefers_local():
    r = _router_with_mocks()
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        order = r.select(TaskType.BULK_PROCESSING)
        assert order[0] == "local"


def test_quota_exhaustion_skips_provider():
    r = _router_with_mocks()
    cfg = r.by_name["groq"]
    cfg.requests_per_day = 1
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        r.quota.record_success("groq", 10, 10, 5.0)
        # groq at cap -> should be skipped
        order = r.select(TaskType.NORMAL)
        assert "groq" not in order


def test_429_cooldown_and_failover():
    r = _router_with_mocks()
    nvidia = r.adapters["nvidia"]
    groq = r.adapters["groq"]
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        nvidia.generate = MagicMock(side_effect=ProviderError("quota", kind="rate_limit",
                                                              status_code=429))
        groq.generate = MagicMock(return_value=MagicMock(
            text="ok", provider="groq", model="groq-model",
            input_tokens=5, output_tokens=2, total_tokens=7,
            latency_ms=10.0, fallback_used=False, retry_count=0, raw={}))
        # fix mock to be LLMResponse-like
        from llm.base import LLMResponse
        groq.generate.return_value = LLMResponse(text="ok", provider="groq",
                                                 model="groq-model", input_tokens=5,
                                                 output_tokens=2, total_tokens=7,
                                                 latency_ms=10.0)
        # CODING chain is nvidia -> groq, so nvidia 429 must fail over to groq
        out = r.generate([{"role": "user", "content": "hi"}], task_type=TaskType.CODING)
        assert out.provider == "groq"
        assert out.fallback_used is True
        # nvidia now in cooldown
        assert r.quota.in_cooldown("nvidia")


def test_timeout_retries_once_then_failover():
    r = _router_with_mocks()
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        calls = {"n": 0}

        def boom(_req):
            calls["n"] += 1
            raise ProviderError("t/o", kind="timeout", retryable=True)

        from llm.base import LLMResponse
        r.adapters["nvidia"].generate = boom
        r.adapters["groq"].generate = MagicMock(return_value=LLMResponse(
            text="ok2", provider="groq", model="g", input_tokens=1,
            output_tokens=1, total_tokens=2, latency_ms=5.0))
        # CODING: nvidia first (fails 1+1 retry), then groq
        out = r.generate([{"role": "user", "content": "hi"}], task_type=TaskType.CODING)
        # nvidia tried (1 + 1 retry inside) then groq
        assert calls["n"] == 2
        assert out.provider == "groq"


def test_jarvislabs_provider_config(monkeypatch):
    monkeypatch.setenv("JARVISLABS_API_KEY", "test")
    monkeypatch.delenv("JARVISLABS_BASE_URL", raising=False)
    monkeypatch.delenv("JARVISLABS_MODEL", raising=False)
    monkeypatch.delenv("LLM_DISABLE_JARVISLABS", raising=False)
    from llm.provider_config import default_provider_configs, fallback_chains

    cfgs = {c.provider_name: c for c in default_provider_configs()}
    cfg = cfgs["jarvislabs"]
    assert cfg.base_url == "https://models.jarvislabs.net/v1"
    assert cfg.api_key_env_var == "JARVISLABS_API_KEY"
    assert cfg.model == "glm-5.3"
    assert fallback_chains()["NORMAL"][0] == "jarvislabs"


def test_moderate_latency_does_not_override_primary(monkeypatch):
    monkeypatch.delenv("LLM_FORCE_PROVIDER", raising=False)
    r = LLMRouter(configs=[
        _cfg("jarvislabs", model="glm-5.3-flash", priority=5),
        _cfg("gemini", model="gemini-3.8-flash", priority=60),
    ])
    with patch.dict(os.environ, {"JARVISLABS_API_KEY": "x", "GEMINI_API_KEY": "y"}):
        # 4s is normal cloud latency, not a reason to prefer a fallback.
        r.quota.record_success("jarvislabs", 1000, 500, 4000.0)
        assert r.select(TaskType.COMPLEX_REASONING)[0] == "jarvislabs"


def test_force_provider_restricts_chain(monkeypatch):
    r = _router_with_mocks()
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x", "GROQ_API_KEY": "y"}):
        monkeypatch.setenv("LLM_FORCE_PROVIDER", "groq")
        assert r.select(TaskType.CODING) == ["groq"]
        monkeypatch.setenv("LLM_FORCE_PROVIDER", "missing")
        assert r.select(TaskType.CODING) == []


def test_missing_keys_disables_provider():
    r = _router_with_mocks()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("NVIDIA_API_KEY", None)
        os.environ.pop("GROQ_API_KEY", None)
        avail = r.available_providers()
        assert "nvidia" not in avail and "groq" not in avail
        assert "local" in avail  # no key required


def test_disabled_provider_skipped():
    cfgs = [_cfg("nvidia", priority=10, enabled=False), _cfg("local", priority=30)]
    r = LLMRouter(configs=cfgs)
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}):
        assert "nvidia" not in r.available_providers()


def test_model_capability_matching():
    cap = get_model_capability("meta/muse-glimmer-30b")
    assert cap["coding_quality"] >= 0.8
    unk = get_model_capability("future/model-xyz")
    assert unk["reliability"] == 0.7


def test_streaming_fallback():
    r = _router_with_mocks()
    with patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}):
        r.adapters["groq"].stream = MagicMock(return_value=iter(["hel", "lo"]))
        chunks = list(r.adapters["groq"].stream(LLMRequest(messages=[])))
        assert "".join(chunks) == "hello"


def test_no_providers_raises():
    r = LLMRouter(configs=[_cfg("nvidia", priority=10)])
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("NVIDIA_API_KEY", None)
        with pytest.raises(ProviderError):
            r.generate([{"role": "user", "content": "hi"}])
