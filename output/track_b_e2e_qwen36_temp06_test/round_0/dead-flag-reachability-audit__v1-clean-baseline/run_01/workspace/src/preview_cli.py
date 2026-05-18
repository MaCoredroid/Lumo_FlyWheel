from __future__ import annotations

import json
import os

from preview.config import load_preview_env
from preview.service import build_preview_plan


def main() -> int:
    config = load_preview_env(os.environ)
    print(json.dumps(build_preview_plan(config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
