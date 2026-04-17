from __future__ import annotations

GATE_STEPS = [
    {"label": "queue-check", "required": True, "lane": "dispatch"},
    {"label": "ledger-check", "required": True, "lane": "settlement"},
    {"label": "receipt__audit", "required": False, "lane": "receipts"},
]


def active_labels(*, include_optional: bool = False) -> list[str]:
    return [
        step["label"]
        for step in GATE_STEPS
        if step["required"] or include_optional
    ]
