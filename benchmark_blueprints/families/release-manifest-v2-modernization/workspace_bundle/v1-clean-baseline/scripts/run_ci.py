import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text()
    manifest = (root / "release" / "manifest.v2.toml").read_text()
    if "target_environment" in workflow and 'target_environment = "staging"' in manifest:
        print("dry-run ok")
        return 0
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
