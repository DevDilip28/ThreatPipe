"""Provider factory — only configured providers are instantiated."""

from __future__ import annotations

from typing import Dict, List

from ..base import ProviderAdapter, ProviderConfig
from .cloudflare import CloudflareProvider
from .openai_compat import OpenAICompatProvider


def build_adapters(configs: List[ProviderConfig]) -> Dict[str, ProviderAdapter]:
    out: Dict[str, ProviderAdapter] = {}
    for cfg in configs:
        if cfg.provider_name.lower() == "cloudflare":
            out[cfg.provider_name] = CloudflareProvider(cfg)
        else:
            # jarvislabs, nvidia, groq, local, ollama, opencode, gemini — all OpenAI-compat
            out[cfg.provider_name] = OpenAICompatProvider(cfg)
    return out
