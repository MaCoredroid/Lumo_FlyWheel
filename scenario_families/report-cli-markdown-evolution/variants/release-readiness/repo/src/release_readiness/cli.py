"""CLI entrypoint for the release-readiness report."""
from __future__ import annotations

import argparse
import json
import sys

from release_readiness.adapters.env_source import EnvSource
from release_readiness.adapters.fs_source import FsSource
from release_readiness.config import Settings
from release_readiness.core.aggregation import build_report
from release_readiness.core.model import Section, sections_from_iterable
from release_readiness.renderers.registry import get_registry


def _build_parser() -> argparse.ArgumentParser:
    registry = get_registry()
    formats = registry.available_formats()
    parser = argparse.ArgumentParser(prog="release-readiness")
    parser.add_argument(
        "--format",
        choices=formats,
        default=formats[0] if formats else "json",
        help=f"Output format (available: {', '.join(formats)})",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read records as JSON from stdin instead of the configured source.",
    )
    return parser


def render_report_from_sections(
    sections: tuple[Section, ...],
    *,
    fmt: str,
    title: str = "Release Readiness Report",
    known_owners: tuple[str, ...] = (),
) -> str:
    """Programmatic entrypoint used by tests and by other internal callers."""
    report = build_report(title=title, sections=sections, known_owners=known_owners)
    renderer = get_registry().get(fmt)
    return renderer.render(report)


def main(argv: list[str] | None = None) -> str:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = Settings()

    if args.stdin:
        data = json.loads(sys.stdin.read())
        sections = sections_from_iterable(data)
        known_owners: tuple[str, ...] = ()
    elif settings.source == "fs":
        sections = FsSource(settings.fs_path).load()
        known_owners = ()
    else:
        env = EnvSource()
        sections = env.load()
        known_owners = env.known_owners()

    return render_report_from_sections(
        sections,
        fmt=args.format,
        title=settings.title,
        known_owners=known_owners,
    )


if __name__ == "__main__":  # pragma: no cover
    print(main())
