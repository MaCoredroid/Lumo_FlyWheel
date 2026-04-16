from __future__ import annotations

CHECKLIST = [{'label': 'ranking-check', 'required': True}, {'label': 'fixture-check', 'required': True}]


def run_checks() -> list[str]:
    return [item["label"] for item in CHECKLIST if item["required"]]
