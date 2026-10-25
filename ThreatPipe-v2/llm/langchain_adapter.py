"""ThreatPipe v2 — LangChain compatibility shim.

Existing code uses:
    llm = get_llm()
    structured = llm.with_structured_output(PydanticModel)
    result = chain.invoke({...})

This shim keeps that call pattern but routes through LLMRouter.
If langchain_core is installed we still avoid provider-specific imports;
otherwise we provide a minimal prompt|llm chain implementation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from .base import TaskType
from .router import get_router


class RouterChatModel:
    """Minimal Chat model speaking to LLMRouter, with with_structured_output()."""

    def __init__(self, task_type: TaskType = TaskType.NORMAL, temperature: float = 0.0,
                 max_tokens: int = 2000):
        self.task_type = task_type
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages) -> Any:
        # messages: list of (role, content) tuples or langchain prompt value
        norm = _normalize_messages(messages)
        resp = get_router().generate(norm, task_type=self.task_type,
                                     temperature=self.temperature,
                                     max_tokens=self.max_tokens)
        return _SimpleAIMessage(resp.text, resp)

    def with_structured_output(self, schema: Type[BaseModel]):
        parent = self
        schema_msg = {"role": "system", "content": _schema_instruction(schema)}

        class Structured:
            def invoke(self, inputs: Dict[str, Any] | Any) -> BaseModel:
                # inputs is usually a dict for prompt variables; we need rendered prompt.
                # If caller used prompt | structured, inputs is already rendered string/dict.
                # Here we handle dict -> JSON instruction fallback.
                if isinstance(inputs, dict):
                    # Build a compact instruction; DO NOT log full prompt contents elsewhere.
                    content = json.dumps(inputs, default=str)[:6000]
                    msgs = [schema_msg,
                            {"role": "user", "content": f"Input:\n{content}"}]
                else:
                    msgs = [schema_msg] + _normalize_messages(inputs)
                # Ask router for JSON; temperature 0 for determinism
                resp = get_router().generate(msgs, task_type=parent.task_type,
                                             temperature=0.0, max_tokens=parent.max_tokens)
                return _parse_structured(resp.text, schema)

            # Make coercible to langchain Runnable: `prompt | structured`
            # langchain's coerce_to_runnable accepts Runnable | callable | dict.
            def __call__(self, inputs: Any) -> BaseModel:
                return self.invoke(inputs)

        return Structured()


class _SimpleAIMessage:
    def __init__(self, content: str, resp=None):
        self.content = content
        self._resp = resp


def _normalize_messages(m) -> List[Dict[str, str]]:
    # langchain ChatPromptValue -> .to_messages()
    try:
        if hasattr(m, "to_messages"):
            out = []
            for x in m.to_messages():
                role = getattr(x, "type", getattr(x, "role", "user"))
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                out.append({"role": role, "content": getattr(x, "content", str(x))})
            return out
    except Exception:
        pass
    if isinstance(m, list):
        out = []
        for x in m:
            if isinstance(x, (tuple, list)) and len(x) == 2:
                r, c = x
                out.append({"role": "user" if r == "human" else r, "content": str(c)})
            elif isinstance(x, dict) and "content" in x:
                out.append({"role": x.get("role", "user"), "content": str(x["content"])})
            else:
                out.append({"role": "user", "content": str(x)})
        return out
    return [{"role": "user", "content": str(m)}]


def _schema_instruction(schema: Type[BaseModel]) -> str:
    """Explicit schema hint; small models need field names + descriptions."""
    try:
        spec = json.dumps(schema.model_json_schema(), indent=2)
    except Exception:
        spec = "{}"
    return (
        f"Return ONLY one JSON object matching the {schema.__name__} schema. "
        "No markdown fences, no prose, no extra keys. Use the exact field names.\n"
        f"JSON Schema:\n{spec[:5000]}"
    )


def _parse_structured(text: str, schema: Type[BaseModel]) -> BaseModel:
    # Robust JSON extraction (code fences, prefixes)
    t = text.strip()
    if "```" in t:
        try:
            t = t.split("```")[1]
            if t.startswith("json"):
                t = t[4:]
            t = t.strip()
        except Exception:
            pass
    # find first { ... last }
    s, e = t.find("{"), t.rfind("}")
    if s != -1 and e != -1 and e > s:
        t = t[s:e + 1]
    try:
        data = json.loads(t)
        # Repair: Stage-1 models often omit `severity` — default it.
        if isinstance(data, dict) and "suspicious" in data and isinstance(data["suspicious"], list):
            fixed = []
            for item in data["suspicious"]:
                if isinstance(item, dict):
                    item = dict(item)
                    item.setdefault("severity", "MEDIUM")
                    item.setdefault("reason", "flagged by triage")
                    # coerce line_number to int; skip items without a usable number
                    raw_num = str(item.get("line_number", ""))
                    digits = re.search(r"\d+", raw_num)
                    if not digits:
                        continue
                    item["line_number"] = int(digits.group())
                    fixed.append(item)
            data["suspicious"] = fixed
            data.setdefault("window_summary", "")
        return schema(**data)
    except Exception:
        # last resort: try raw text as explanation field if schema allows
        try:
            # build minimal valid instance per known schemas
            fields = schema.model_fields.keys()
            fb: Dict[str, Any] = {}
            if "suspicious" in fields:
                fb = {"suspicious": [], "window_summary": text[:500]}
            elif "status" in fields:
                fb = {"status": "AMBIGUOUS", "explanation": text[:1000],
                      "verdict": "SUSPICIOUS", "confidence": 0.3}
            elif "campaign_type" in fields:
                fb = {"campaign_type": "opportunistic", "active_ips": 1,
                      "highest_stage": "RECON", "primary_vector": "unknown",
                      "kill_chain_stage": "RECON", "summary": text[:1000],
                      "ip_alerts": [], "immediate_actions": []}
            else:
                raise
            return schema(**fb)
        except Exception:
            raise ValueError(f"structured parse failed: {text[:300]}")


def get_chat_model(task_type: TaskType = TaskType.NORMAL, temperature: float = 0.0,
                   max_tokens: int = 2000) -> RouterChatModel:
    return RouterChatModel(task_type=task_type, temperature=temperature, max_tokens=max_tokens)
