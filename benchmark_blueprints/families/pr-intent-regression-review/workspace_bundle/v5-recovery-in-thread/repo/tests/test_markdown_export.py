from release_readiness.export import export_report


def test_markdown_export_uses_snapshot() -> None:
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output="markdown")
    assert rendered.startswith("# Release Readiness")


def test_markdown_snapshot_stays_stable() -> None:
    fixture = open("repo/tests/fixtures/release_readiness.md").read().strip()
    report = {"version": 1, "ready": True, "services": ["api", "worker"]}
    rendered = export_report(report, output="markdown")
    assert rendered.strip() == fixture
