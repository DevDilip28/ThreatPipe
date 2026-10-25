"""
ThreatPipe v2 — Provider configuration loader.

- Reads env vars; enables only configured providers.
- Limits are configurable via config.yaml (llm_routing) + env overrides.
- Never stores API keys, only env-var NAMES.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import yaml

from .base import ProviderConfig


def _yaml_routing_cfg() -> Dict:
    for cand in [Path(__file__).parent.parent / "config.yaml", Path("config.yaml")]:
        try:
            if cand.exists():
                with open(cand) as f:
                    data = yaml.safe_load(f) or {}
                return data.get("llm_routing", {}) or {}
        except Exception:
            continue
    return {}


def _int_env(name: str, default: int) -> int:
    try:
        v = os.getenv(name)
        return int(v) if v else default
    except Exception:
        return default


def default_provider_configs() -> List[ProviderConfig]:
    """
    Priority defaults match user's setup:
      Coding: Jarvislabs -> NVIDIA -> Groq -> Ollama/local
      Simple: Jarvislabs -> Groq/local -> NVIDIA
      Complex: Jarvislabs -> NVIDIA -> Groq/OpenCode -> local
      Bulk: local/Ollama first unless Jarvislabs is forced
    Exact order is still reconfigurable via config.yaml fallback_chains.
    """
    y = _yaml_routing_cfg()
    prov_over = (y.get("providers") or {}) if isinstance(y, dict) else {}

    def ov(name: str, field: str, default):
        try:
            return (prov_over.get(name) or {}).get(field, default)
        except Exception:
            return default

    cfgs = [
        ProviderConfig(
            # Jarvislabs Token Factory / Serverless — OpenAI-compatible.
            # Key: https://jarvislabs.ai/settings/api-keys
            # Hosted: https://models.jarvislabs.net/v1
            # Serverless: https://serverlessn.jarvislabs.net/openai/<deployment_id>/v1
            provider_name="jarvislabs",
            base_url=os.getenv("JARVISLABS_BASE_URL", "https://models.jarvislabs.net/v1"),
            model=os.getenv("JARVISLABS_MODEL", "glm-5.3"),
            api_key_env_var="JARVISLABS_API_KEY",
            priority=ov("jarvislabs", "priority", 5),
            capabilities=["chat", "tools", "streaming", "reasoning", "coding", "bulk"],
            requests_per_minute=ov("jarvislabs", "requests_per_minute",
                                   _int_env("JARVISLABS_RPM", 60)),
            requests_per_day=ov("jarvislabs", "requests_per_day",
                                _int_env("JARVISLABS_RPD", 5000)),
            tokens_per_minute=ov("jarvislabs", "tokens_per_minute",
                                 _int_env("JARVISLABS_TPM", 200000)),
            tokens_per_day=ov("jarvislabs", "tokens_per_day",
                              _int_env("JARVISLABS_TPD", 5000000)),
            cost=0.0,
            context_window=ov("jarvislabs", "context_window",
                              _int_env("JARVISLABS_CONTEXT_WINDOW", 131072)),
            supports_tools=os.getenv("JARVISLABS_SUPPORTS_TOOLS", "1") != "0",
            supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_JARVISLABS", "") != "1",
        ),
        ProviderConfig(
            provider_name="nvidia",
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            model=os.getenv("NVIDIA_MODEL", "meta/muse-glimmer-30b"),
            api_key_env_var="NVIDIA_API_KEY",
            priority=ov("nvidia", "priority", 10),
            capabilities=["chat", "tools", "streaming", "reasoning", "coding"],
            requests_per_minute=ov("nvidia", "requests_per_minute",
                                   _int_env("NVIDIA_RPM", 40)),
            requests_per_day=ov("nvidia", "requests_per_day",
                                _int_env("NVIDIA_RPD", 1000)),
            tokens_per_minute=ov("nvidia", "tokens_per_minute",
                                 _int_env("NVIDIA_TPM", 60000)),
            tokens_per_day=ov("nvidia", "tokens_per_day",
                              _int_env("NVIDIA_TPD", 500000)),
            cost=0.0, context_window=131072,
            supports_tools=True, supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_NVIDIA", "") != "1",
            extra_models=[m for m in [os.getenv("NVIDIA_MODEL2", "google/gemma-4-31b-it")] if m],
        ),
        ProviderConfig(
            provider_name="groq",
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key_env_var="GROQ_API_KEY",
            priority=ov("groq", "priority", 20),
            capabilities=["chat", "tools", "streaming", "fast"],
            requests_per_minute=ov("groq", "requests_per_minute",
                                   _int_env("GROQ_RPM", 30)),
            requests_per_day=ov("groq", "requests_per_day",
                                _int_env("GROQ_RPD", 5000)),
            tokens_per_minute=ov("groq", "tokens_per_minute",
                                 _int_env("GROQ_TPM", 60000)),
            tokens_per_day=ov("groq", "tokens_per_day",
                              _int_env("GROQ_TPD", 500000)),
            cost=0.0, context_window=32768,
            supports_tools=True, supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_GROQ", "") != "1",
        ),
        ProviderConfig(
            # llama.cpp OpenAI-compat server (user: LLM_BASE_URL=http://localhost:8080/v1)
            provider_name="local",
            base_url=os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:8080/v1")),
            model=os.getenv("LLM_MODEL_NAME", os.getenv("OLLAMA_MODEL", "gemma-4-E2B-it-Q4_K_M.gguf")),
            api_key_env_var="LLM_API_KEY",  # unused; no auth needed
            priority=ov("local", "priority", 30),
            capabilities=["chat", "streaming", "offline", "bulk"],
            requests_per_minute=ov("local", "requests_per_minute", 120),
            requests_per_day=ov("local", "requests_per_day", 100000),
            tokens_per_minute=ov("local", "tokens_per_minute", 200000),
            tokens_per_day=ov("local", "tokens_per_day", 10000000),
            cost=0.0, context_window=8192,
            supports_tools=False, supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_LOCAL", "") != "1",
        ),
        ProviderConfig(
            provider_name="ollama",
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/v1",
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            api_key_env_var="OLLAMA_API_KEY",
            priority=ov("ollama", "priority", 40),
            capabilities=["chat", "streaming", "offline", "bulk"],
            requests_per_minute=120, requests_per_day=100000,
            tokens_per_minute=200000, tokens_per_day=10000000,
            cost=0.0, context_window=8192,
            supports_tools=False, supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_OLLAMA", "") != "1",
        ),
        ProviderConfig(
            # OpenCode Zen gateway — OpenAI-compatible. Optional.
            # Curl-verified 2026-10-08: only space-bunny-free works via direct
            # /chat/completions API. Other *-free models return FreeTierError
            # ("can only be used from within OpenCode"). Default accordingly.
            provider_name="opencode",
            base_url=os.getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1"),
            model=os.getenv("OPENCODE_MODEL", "space-bunny-free"),
            api_key_env_var="OPENCODE_API_KEY",
            priority=ov("opencode", "priority", 50),
            capabilities=["chat", "tools", "streaming"],
            requests_per_minute=ov("opencode", "requests_per_minute", 30),
            requests_per_day=ov("opencode", "requests_per_day", 1000),
            tokens_per_minute=ov("opencode", "tokens_per_minute", 60000),
            tokens_per_day=ov("opencode", "tokens_per_day", 500000),
            cost=0.0, context_window=65536,
            supports_tools=True, supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_OPENCODE", "") != "1",
        ),
        ProviderConfig(
            provider_name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            api_key_env_var="GEMINI_API_KEY",
            priority=ov("gemini", "priority", 60),
            capabilities=["chat", "tools", "long_context"],
            requests_per_minute=15, requests_per_day=1500,
            tokens_per_minute=1000000, tokens_per_day=5000000,
            cost=0.0, context_window=1048576,
            supports_tools=True, supports_streaming=True,
            enabled=os.getenv("LLM_DISABLE_GEMINI", "") != "1",
        ),
        ProviderConfig(
            provider_name="cloudflare",
            base_url=os.getenv("CLOUDFLARE_BASE_URL", "https://api.cloudflare.com/client/v4/accounts"),
            model=os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.1-8b-instruct"),
            api_key_env_var="CLOUDFLARE_API_KEY",
            priority=ov("cloudflare", "priority", 70),
            capabilities=["chat", "bulk"],
            requests_per_minute=20, requests_per_day=10000,
            tokens_per_minute=30000, tokens_per_day=300000,
            cost=0.0, context_window=8192,
            supports_tools=False, supports_streaming=False,
            enabled=os.getenv("LLM_DISABLE_CLOUDFLARE", "") != "1",
        ),
    ]
    return cfgs


def fallback_chains() -> Dict[str, List[str]]:
    """Configurable fallback order per task. Overridable via config.yaml."""
    y = _yaml_routing_cfg()
    chains = (y.get("fallback_chains") or {}) if isinstance(y, dict) else {}
    defaults = {
        "CODING": ["jarvislabs", "nvidia", "groq", "opencode", "local", "ollama"],
        "COMPLEX_REASONING": ["jarvislabs", "nvidia", "groq", "opencode", "local", "ollama"],
        "TOOL_CALLING": ["jarvislabs", "groq", "nvidia", "opencode", "gemini", "local"],
        "NORMAL": ["jarvislabs", "groq", "nvidia", "opencode", "gemini", "local", "ollama"],
        "SIMPLE": ["jarvislabs", "groq", "opencode", "local", "nvidia", "cloudflare", "ollama"],
        "BULK_PROCESSING": ["jarvislabs", "opencode", "groq", "nvidia", "local", "ollama", "cloudflare"],
        "LONG_CONTEXT": ["jarvislabs", "gemini", "nvidia", "groq", "opencode", "local"],
    }
    defaults.update(chains)
    return defaults


def routing_mode() -> str:
    y = _yaml_routing_cfg()
    m = os.getenv("LLM_ROUTING_MODE", (y.get("mode") or "HYBRID"))
    m = str(m).upper()
    return m if m in ("QUALITY_FIRST", "FREE_QUOTA_FIRST", "HYBRID") else "HYBRID"


def force_provider() -> str:
    """Optional single-provider override.

    Use when a paid subscription should handle every task, e.g.
    LLM_FORCE_PROVIDER=jarvislabs. Empty means normal scored routing.
    """
    y = _yaml_routing_cfg()
    v = os.getenv("LLM_FORCE_PROVIDER", "")
    if not v and isinstance(y, dict):
        v = y.get("force_provider") or ""
    return str(v).strip()


def call_delay_s(default: float = 5.0) -> float:
    """Per-call throttle to reduce provider rate-limit blocking.

    Configurable via env LLM_CALL_DELAY_S or config.yaml llm_routing.call_delay_s.
    Set 0 to disable (used by tests).
    """
    try:
        env = os.getenv("LLM_CALL_DELAY_S")
        if env is not None and env != "":
            return max(0.0, float(env))
    except Exception:
        pass
    try:
        y = _yaml_routing_cfg()
        v = (y.get("call_delay_s") if isinstance(y, dict) else None)
        if v is not None:
            return max(0.0, float(v))
    except Exception:
        pass
    return default
