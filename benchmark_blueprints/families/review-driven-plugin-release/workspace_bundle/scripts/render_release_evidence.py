from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from drive_brief.settings import render_settings_panel

PLUGIN_PATH = ROOT / ".codex-plugin" / "plugin.json"
CHECKLIST_PATH = ROOT / "docs" / "release_checklist.md"
NOTES_PATH = ROOT / "docs" / "release_notes.md"
THREADS_PATH = ROOT / "review" / "unresolved_threads.json"
REVIEW_REPLY_PATH = ROOT / "artifacts" / "review_reply.json"
EVIDENCE_PATH = ROOT / "artifacts" / "release_evidence.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def build_release_evidence() -> dict:
    plugin = _read_json(PLUGIN_PATH)
    threads = _read_json(THREADS_PATH)
    review_reply = _read_json(REVIEW_REPLY_PATH)

    screenshots = {
        "before": "artifacts/screenshots/settings-panel-before.png",
        "expected": "artifacts/screenshots/settings-panel-expected.png",
    }

    return {
        "plugin_id": plugin["id"],
        "manifest_backward_compatibility": {
            "field": "name",
            "value": plugin.get("name"),
            "remove_after": plugin["compatibility"]["deprecated_fields"]["name"][
                "remove_after"
            ],
        },
        "settings_panel_states": {
            "unset_optional_value": {
                "html": render_settings_panel(),
                "toggle_visible": "Connector fallback" in render_settings_panel(),
            },
            "configured_optional_value": {
                "html": render_settings_panel("shared-drive"),
                "toggle_visible": "Connector fallback"
                in render_settings_panel("shared-drive"),
            },
        },
        "docs": {
            "release_notes": NOTES_PATH.relative_to(ROOT).as_posix(),
            "release_checklist": CHECKLIST_PATH.relative_to(ROOT).as_posix(),
        },
        "review": {
            "actionable_thread_ids": [
                item["id"] for item in threads["unresolved"]
            ]
            + [threads["screenshot_note"]["id"]],
            "reply_ids": [
                item["id"] for item in review_reply["resolved_threads"]
            ],
            "reply_path": REVIEW_REPLY_PATH.relative_to(ROOT).as_posix(),
        },
        "screenshots": screenshots,
    }


def main() -> None:
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(build_release_evidence(), indent=2) + "\n")


if __name__ == "__main__":
    main()
