"""
ThreatPipe v2 — Model capability registry.

Do NOT choose models only by parameter count.
Prefer MoE when quality/latency tradeoff is better.
Easy to update when providers change models.
"""

from __future__ import annotations

from typing import Dict

# Scores are 0.0..1.0 relative guides, not benchmarks.
# free_tier_status: "free" | "freemium" | "paid" | "local" (no quota)
MODEL_REGISTRY: Dict[str, Dict] = {
    # --- Jarvislabs Token Factory / Serverless (paid, OpenAI-compatible) ---
    "glm-5.3": {
        "provider": "jarvislabs",
        "reasoning_quality": 0.90, "coding_quality": 0.88, "tool_use_quality": 0.82,
        "speed": 0.72, "context_length": 131072, "reliability": 0.88,
        "estimated_cost": 0.0, "free_tier_status": "paid",
        "notes": "Strong paid default for COMPLEX_REASONING/CODING on Jarvislabs.",
    },
    "glm-5.3-flash": {
        "provider": "jarvislabs",
        "reasoning_quality": 0.80, "coding_quality": 0.78, "tool_use_quality": 0.75,
        "speed": 0.92, "context_length": 131072, "reliability": 0.85,
        "estimated_cost": 0.0, "free_tier_status": "paid",
        "notes": "Cheaper/faster Jarvislabs option for SIMPLE/BULK/NORMAL.",
    },
    "deepseek-v4.1-flash-1017": {
        "provider": "jarvislabs",
        "reasoning_quality": 0.84, "coding_quality": 0.86, "tool_use_quality": 0.80,
        "speed": 0.88, "context_length": 131072, "reliability": 0.84,
        "estimated_cost": 0.0, "free_tier_status": "paid",
        "notes": "Balanced Jarvislabs model for coding + tool-style JSON work.",
    },
    "deepseek-v4-flash-0731": {
        "provider": "jarvislabs",
        "reasoning_quality": 0.78, "coding_quality": 0.80, "tool_use_quality": 0.75,
        "speed": 0.90, "context_length": 131072, "reliability": 0.82,
        "estimated_cost": 0.0, "free_tier_status": "paid",
        "notes": "Fast Jarvislabs model for bulk triage.",
    },
    # --- Groq (fast, generous free tier) ---
    "openai/gpt-oss-20b": {
        "provider": "groq",
        "reasoning_quality": 0.65, "coding_quality": 0.70, "tool_use_quality": 0.75,
        "speed": 0.95, "context_length": 32768, "reliability": 0.85,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "MoE-ish OSS; great for SIMPLE/BULK + tool calling on Groq.",
    },
    "llama-3.3-70b-versatile": {
        "provider": "groq",
        "reasoning_quality": 0.75, "coding_quality": 0.75, "tool_use_quality": 0.80,
        "speed": 0.90, "context_length": 32768, "reliability": 0.85,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Balanced default for NORMAL/TOOL_CALLING on Groq.",
    },
    # --- NVIDIA NIM (free build.nvidia.com credits) ---
    "meta/muse-glimmer-30b": {
        "provider": "nvidia",
        "reasoning_quality": 0.85, "coding_quality": 0.85, "tool_use_quality": 0.80,
        "speed": 0.70, "context_length": 131072, "reliability": 0.80,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Primary per user config; strong CODING/COMPLEX_REASONING.",
    },
    "google/gemma-4-31b-it": {
        "provider": "nvidia",
        "reasoning_quality": 0.78, "coding_quality": 0.75, "tool_use_quality": 0.70,
        "speed": 0.75, "context_length": 32768, "reliability": 0.78,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Secondary NVIDIA model (NVIDIA_MODEL2).",
    },
    # --- Local (llama.cpp server + Ollama, no quota) ---
    "gemma-4-E2B-it-Q4_K_M.gguf": {
        "provider": "local",
        "reasoning_quality": 0.55, "coding_quality": 0.55, "tool_use_quality": 0.45,
        "speed": 0.60, "context_length": 8192, "reliability": 0.90,
        "estimated_cost": 0.0, "free_tier_status": "local",
        "notes": "Local GGUF via llama.cpp server at LLM_BASE_URL. Prefer for BULK/offline.",
    },
    "llama3.1:8b": {
        "provider": "ollama",
        "reasoning_quality": 0.55, "coding_quality": 0.60, "tool_use_quality": 0.50,
        "speed": 0.65, "context_length": 8192, "reliability": 0.90,
        "estimated_cost": 0.0, "free_tier_status": "local",
        "notes": "Ollama fallback; no API quota.",
    },
    # --- OpenCode Zen gateway (optional, only if OPENCODE_API_KEY set) ---
    # Curl-verified: space-bunny-free works via /chat/completions + JSON.
    # big-pickle / longcat / mimo / ling / nemotron / muse-spark-free return
    # FreeTierError via direct API (OpenCode-client-only).
    "space-bunny-free": {
        "provider": "opencode",
        "reasoning_quality": 0.68, "coding_quality": 0.65, "tool_use_quality": 0.60,
        "speed": 0.80, "context_length": 32768, "reliability": 0.82,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Zen free stealth model; verified /chat/completions + JSON.",
    },
    "big-pickle": {
        "provider": "opencode",
        "reasoning_quality": 0.70, "coding_quality": 0.72, "tool_use_quality": 0.65,
        "speed": 0.80, "context_length": 32768, "reliability": 0.80,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Zen free stealth model; OpenAI-compatible /chat/completions.",
    },
    "opencode/muse-spark-1.3-contributor-free": {
        "provider": "opencode",
        "reasoning_quality": 0.70, "coding_quality": 0.72, "tool_use_quality": 0.70,
        "speed": 0.75, "context_length": 65536, "reliability": 0.80,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Zen free contributor model; OpenAI-compatible /responses via gateway.",
    },
    # --- Generic placeholders (enabled only if keys appear later) ---
    "gemini-3.8-flash": {
        "provider": "gemini",
        "reasoning_quality": 0.78, "coding_quality": 0.72, "tool_use_quality": 0.75,
        "speed": 0.85, "context_length": 1048576, "reliability": 0.85,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Large context; LONG_CONTEXT tasks when GEMINI_API_KEY is set. 2.0 retired, use 3.8.",
    },
    "gemini-2.0-flash": {
        "provider": "gemini",
        "reasoning_quality": 0.75, "coding_quality": 0.70, "tool_use_quality": 0.75,
        "speed": 0.85, "context_length": 1048576, "reliability": 0.85,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Large context; LONG_CONTEXT tasks when GEMINI_API_KEY is set.",
    },
}


def get_model_capability(model_id: str) -> Dict:
    """Return capability dict; unknown models get a neutral default."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]
    # default for user-supplied future models
    return {
        "provider": "unknown",
        "reasoning_quality": 0.6, "coding_quality": 0.6, "tool_use_quality": 0.6,
        "speed": 0.7, "context_length": 8192, "reliability": 0.7,
        "estimated_cost": 0.0, "free_tier_status": "free",
        "notes": "Unlisted model — neutral defaults.",
    }


def list_models() -> Dict[str, Dict]:
    return dict(MODEL_REGISTRY)
