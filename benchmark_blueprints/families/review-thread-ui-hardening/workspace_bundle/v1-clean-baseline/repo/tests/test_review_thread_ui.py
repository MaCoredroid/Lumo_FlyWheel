from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
COMPONENT = ROOT / "repo" / "src" / "components" / "ReviewThreadCard.tsx"
STYLE = ROOT / "repo" / "src" / "styles" / "review-thread.css"
CONFIG = ROOT / "repo" / "config" / "snapshot-viewports.json"


def main() -> int:
    component = COMPONENT.read_text()
    config = json.loads(CONFIG.read_text())

    assert 'data-control="reply-thread-menu"' in component
    assert 'data-control="pin-thread-menu"' in component
    assert "renderReviewThreadCards" in component
    assert isinstance(config["viewports"], list)
    assert isinstance(config["scenarios"], list)
    assert len({item["id"] for item in config["viewports"]}) == len(config["viewports"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
