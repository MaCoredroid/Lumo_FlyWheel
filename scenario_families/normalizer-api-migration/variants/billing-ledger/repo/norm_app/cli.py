from __future__ import annotations

from norm_app.assembler import compile_payload
from norm_app.router import route_for


SAMPLE = {'title': 'Invoice Retry', 'owner': 'finance', 'region': 'eu-west'}


def preview() -> dict[str, str]:
    compiled = compile_payload(SAMPLE)
    compiled["route"] = route_for(SAMPLE)
    return compiled
