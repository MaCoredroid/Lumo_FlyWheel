from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "repo" / "src" / "components" / "ReviewThreadCard.tsx"
STYLE = ROOT / "repo" / "src" / "styles" / "review-thread.css"
CONFIG = ROOT / "repo" / "config" / "snapshot-viewports.json"


def main() -> int:
    component = COMPONENT.read_text()
    style = STYLE.read_text()
    config = json.loads(CONFIG.read_text())

    assert 'data-control="reply-thread-menu"' in component
    assert 'aria-label="' in component
    assert "flex-wrap: wrap" in style
    assert "overflow: hidden" not in style
    assert any(item["route"].startswith("/pull/241/") for item in config["scenarios"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
