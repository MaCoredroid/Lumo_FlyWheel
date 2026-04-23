#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path


AGENT_WS = Path(os.environ["AGENT_WS"])
GOLD_FILE = Path(os.environ["GOLD_FILE"])


def _button_has_label(component: str, control_id: str) -> bool:
    pattern = rf'<button(?=[^>]*data-control="{re.escape(control_id)}")(?=[^>]*aria-label="[^"]+")[^>]*>'
    return re.search(pattern, component, re.S) is not None


def main() -> int:
    gold = json.loads(GOLD_FILE.read_text())
    component = (AGENT_WS / gold["component_file"]).read_text(encoding="utf-8", errors="replace")
    style = (AGENT_WS / gold["style_file"]).read_text(encoding="utf-8", errors="replace")
    config = json.loads((AGENT_WS / gold["config_file"]).read_text())

    reply_text = ""
    reply_path = AGENT_WS / gold["reply_file"]
    if reply_path.exists():
        reply_text = reply_path.read_text(encoding="utf-8", errors="replace")

    evidence_text = ""
    evidence_path = AGENT_WS / gold["evidence_file"]
    if evidence_path.exists():
        evidence_text = evidence_path.read_text(encoding="utf-8", errors="replace")

    expected_viewport = gold["viewport"]
    expected_route = gold["route"]

    viewports = {item["id"]: item for item in config.get("viewports", []) if isinstance(item, dict) and "id" in item}
    scenarios = config.get("scenarios", [])
    viewport_present = (
        expected_viewport["id"] in viewports
        and viewports[expected_viewport["id"]].get("width") == expected_viewport["width"]
        and viewports[expected_viewport["id"]].get("height") == expected_viewport["height"]
    )
    route_mapping_present = any(
        isinstance(item, dict)
        and item.get("route") == expected_route
        and item.get("viewport_id") == expected_viewport["id"]
        for item in scenarios
    )

    replies_match = all(thread_id in reply_text for thread_id in gold["unresolved_thread_ids"])
    resolved_omitted = all(thread_id not in reply_text for thread_id in gold["resolved_thread_ids"])
    evidence_matches = expected_route in evidence_text and gold["viewport_human"] in evidence_text

    required_ack = gold.get("required_acknowledgement", "")
    ack_present = True
    if required_ack:
        needle = required_ack.lower()
        ack_present = needle in reply_text.lower() or needle in evidence_text.lower()

    result = {
        "target_control_named": _button_has_label(component, gold["target_control"]),
        "protected_controls_unchanged": all(
            not _button_has_label(component, control_id)
            for control_id in gold.get("protected_controls", [])
        ),
        "wrap_fix": "flex-wrap: wrap" in style and "white-space: normal" in style,
        "no_clip_fix": all(token not in style for token in ("overflow: hidden", "text-overflow", "line-clamp")),
        "exact_viewport": viewport_present,
        "exact_route_mapping": route_mapping_present,
        "replies_match_threads": replies_match,
        "resolved_threads_omitted": resolved_omitted,
        "evidence_matches_viewport": evidence_matches,
        "required_acknowledgement_present": ack_present,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
