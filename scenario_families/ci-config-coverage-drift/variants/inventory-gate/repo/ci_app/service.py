from __future__ import annotations

CHECKLIST = [{'label': 'schema-check', 'required': True}, {'label': 'render-check', 'required': True}]


def run_checks() -> list[str]:
    return [item["label"] for item in CHECKLIST if item["required"]]
