from __future__ import annotations

CHECKLIST = [{'label': 'queue-check', 'required': True}, {'label': 'ledger-check', 'required': True}]


def run_checks() -> list[str]:
    return [item["label"] for item in CHECKLIST if item["required"]]
