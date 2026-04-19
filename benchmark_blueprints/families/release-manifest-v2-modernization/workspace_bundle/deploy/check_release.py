from pathlib import Path


def staging_smoke_ok() -> bool:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text()
    manifest = (root / "release" / "manifest.v2.toml").read_text()
    config = (root / ".codex" / "config.toml").read_text()
    return (
        "target_environment" in workflow
        and 'target_environment = "staging"' in manifest
        and 'release_entrypoint = "scripts/run_ci.py"' in config
    )


if __name__ == "__main__":
    raise SystemExit(0 if staging_smoke_ok() else 1)
