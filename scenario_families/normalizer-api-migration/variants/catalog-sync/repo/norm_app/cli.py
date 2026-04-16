from __future__ import annotations

from norm_app.assembler import compile_payload
from norm_app.router import route_for


SAMPLE = {'title': 'Missing SKU', 'owner': 'catalog', 'region': 'ap-south'}


def preview() -> dict[str, str]:
    compiled = compile_payload(SAMPLE)
    compiled["route"] = route_for(SAMPLE)
    return compiled
