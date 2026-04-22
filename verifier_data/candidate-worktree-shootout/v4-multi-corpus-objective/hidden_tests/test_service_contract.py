import os
import sys
from pathlib import Path

WS = Path(os.environ["AGENT_WS"])
sys.path.insert(0, str(WS / "src"))

from report_filters.service import compile_filters


def test_non_cli_callers_share_normalization() -> None:
    assert compile_filters(["Ops---Latency__Summary", "API__Errors"]) == [
        "ops latency summary",
        "api errors",
    ]
