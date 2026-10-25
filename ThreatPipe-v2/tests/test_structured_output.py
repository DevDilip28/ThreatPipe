"""Regression tests for the router's structured-output JSON shim."""

import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.langchain_adapter import _parse_structured, _schema_instruction


class Item(BaseModel):
    line_number: int
    reason: str = "flagged"
    severity: str = "MEDIUM"


class Result(BaseModel):
    suspicious: list[Item] = []
    window_summary: str = ""


def test_schema_instruction_contains_field_names():
    hint = _schema_instruction(Result)
    assert "line_number" in hint
    assert "suspicious" in hint
    assert "JSON Schema" in hint


def test_parse_structured_extracts_numeric_line_number():
    text = '{"suspicious": [{"line_number": "Line 12", "reason": "shell", "severity": "HIGH"}], "window_summary": "ok"}'
    out = _parse_structured(text, Result)
    assert out.suspicious[0].line_number == 12
    assert out.suspicious[0].severity == "HIGH"


def test_parse_structured_drops_missing_line_number():
    text = '{"suspicious": [{"reason": "no number", "severity": "HIGH"}], "window_summary": "ok"}'
    out = _parse_structured(text, Result)
    assert out.suspicious == []
