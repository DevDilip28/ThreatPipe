"""
ThreatPipe v2 — `llm` CLI observability.

Commands:
    python -m llm.cli status
    python -m llm.cli models
    python -m llm.cli providers
    python -m llm.cli usage
    python -m llm.cli test <provider>
    python -m llm.cli route <TASK_TYPE>

Also wired as: python llm_cli.py <cmd> (see llm_cli.py shim)
Never prints API keys — only env-var names + masked presence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _mask_env(var: str) -> str:
    v = os.getenv(var, "")
    if not v:
        return "missing"
    if len(v) <= 8:
        return "set (***)"
    return f"set ({v[:3]}…{v[-2:]}, len={len(v)})"


def cmd_status(router):
    print(f"{'provider':<12}{'model':<38}{'enabled':<9}{'health':<9}{'req_today':<10}{'tok_today':<10}{'remain':<8}{'lat_ms':<9}{'fail':<6}")
    print("-" * 115)
    usage = router.get_usage()
    health = {}
    try:
        health = router.health_check()
    except Exception as exc:
        print(f"(health check error: {exc})")
    for name, u in usage.items():
        h = health.get(name, {})
        ok = h.get("ok")
        hstr = "ok" if ok else ("reach" if h.get("reachable") else "down")
        print(f"{name:<12}{str(u.get('model'))[:37]:<38}{str(u.get('enabled')):<9}"
              f"{hstr:<9}{u.get('requests_today', 0):<10}{u.get('estimated_tokens_today', 0):<10}"
              f"{u.get('remaining', 0):<8}{u.get('avg_latency_ms', 0):<9}{u.get('failures', 0):<6}")
    print()
    print(f"routing_mode={router.mode.value}  available={router.available_providers()}")


def cmd_models():
    from .registry import list_models
    for m, c in list_models().items():
        print(f"- {m}  provider={c.get('provider')} ctx={c.get('context_length')} "
              f"reason={c.get('reasoning_quality')} code={c.get('coding_quality')} "
              f"tools={c.get('tool_use_quality')} speed={c.get('speed')} free={c.get('free_tier_status')}")


def cmd_providers(router):
    for c in router.configs:
        print(f"- {c.provider_name}: model={c.model} base={c.base_url}")
        print(f"    key_env={c.api_key_env_var}({_mask_env(c.api_key_env_var)}) "
              f"enabled={c.enabled} configured={router.adapters[c.provider_name].is_configured()}")
        print(f"    rpm={c.requests_per_minute} rpd={c.requests_per_day} "
              f"tpm={c.tokens_per_minute} tpd={c.tokens_per_day} ctx={c.context_window} "
              f"tools={c.supports_tools} stream={c.supports_streaming} pri={c.priority}")


def cmd_usage(router):
    print(json.dumps(router.get_usage(), indent=2))


def cmd_test(router, provider: str):
    from .base import LLMRequest, TaskType
    if provider not in router.adapters:
        print(f"unknown provider: {provider}  known={list(router.adapters)}")
        sys.exit(2)
    a = router.adapters[provider]
    print(f"configured={a.is_configured()}  info={a.get_model_info().get('model')}")
    print(f"health={a.health_check()}")
    if not a.is_configured():
        print("skipped inference (not configured). Set the env key to enable.")
        return
    try:
        # Direct adapter test (bypasses routing) so `test <provider>` really tests it
        req = LLMRequest(messages=[{"role": "user", "content": "Reply with exactly: ok"}],
                         task_type=TaskType.SIMPLE, max_tokens=256)
        r = a.generate(req)
        print(f"generate ok via {r.provider}/{r.model} lat={r.latency_ms:.0f}ms text={r.text[:120]!r}")
    except Exception as exc:
        print(f"generate failed: {type(exc).__name__}: {exc}")


def cmd_route(router, task: str):
    from .base import TaskType
    try:
        tt = TaskType(task.upper())
    except Exception:
        print(f"unknown task {task}; valid={[t.value for t in TaskType]}")
        sys.exit(2)
    for row in router.route_preview(tt):
        print(f"- {row['provider']:<12} {row['model']:<40} remain={row['remaining']} score={row['score']}")


def main(argv=None):
    from .router import get_router
    ap = argparse.ArgumentParser(prog="llm", description="ThreatPipe LLM router observability")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("models")
    sub.add_parser("providers")
    sub.add_parser("usage")
    p = sub.add_parser("test"); p.add_argument("provider")
    p = sub.add_parser("route"); p.add_argument("task_type")
    args = ap.parse_args(argv)
    router = get_router()
    if args.cmd == "status":
        cmd_status(router)
    elif args.cmd == "models":
        cmd_models()
    elif args.cmd == "providers":
        cmd_providers(router)
    elif args.cmd == "usage":
        cmd_usage(router)
    elif args.cmd == "test":
        cmd_test(router, args.provider)
    elif args.cmd == "route":
        cmd_route(router, args.task_type)


if __name__ == "__main__":
    main()
