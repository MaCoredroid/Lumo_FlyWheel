from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.digest_runner import render_digest


def test_render_digest_uses_expected_headings() -> None:
    output = render_digest(
        [
            {
                "date": "2026-04-14",
                "severity": "sev2",
                "service": "search",
                "status": "open",
                "summary": "Index lag",
            }
        ],
        "short",
    )

    assert output.startswith("# Ops Digest\n\n## Summary\n")
    assert "\n## Events\n" in output
    assert "- Total events: 1" in output


def test_cli_command_matches_skill_contract() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/digest_runner.py",
            "--input",
            "fixtures/incidents/sample_events.json",
            "--summary-length",
            "short",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "# Ops Digest" in result.stdout
    assert "## Summary" in result.stdout
    assert "## Events" in result.stdout
