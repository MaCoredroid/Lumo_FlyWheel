import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_release_driver_dry_run():
    result = subprocess.run(
        [sys.executable, "scripts/run_ci.py", "--mode", "release-dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
