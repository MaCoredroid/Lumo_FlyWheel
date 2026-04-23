from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> str:
    proc = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True)
    return proc.stdout


def test_current_cli_help_exposes_generate_and_config() -> None:
    output = run(sys.executable, "src/release_preview/cli.py", "--help")
    assert "generate" in output
    generate_help = run(sys.executable, "src/release_preview/cli.py", "generate", "--help")
    assert "--config" in generate_help
    assert "--settings" not in generate_help


def test_helper_alias_still_exists_for_compatibility() -> None:
    output = run(sys.executable, "scripts/release_preview_helper.py", "build-preview", "--settings", "configs/release_preview.toml")
    assert "deprecated_alias=true" in output
    assert "python src/release_preview/cli.py generate --config configs/release_preview.toml" in output
