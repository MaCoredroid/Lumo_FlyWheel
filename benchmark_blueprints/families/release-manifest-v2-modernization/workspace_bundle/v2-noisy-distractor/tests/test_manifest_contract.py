from pathlib import Path


def test_workflow_and_manifest_use_target_environment():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text()
    manifest = (root / "release" / "manifest.v2.toml").read_text()
    assert "target_environment" in workflow
    assert 'target_environment = "staging"' in manifest
