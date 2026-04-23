#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", required=True)
    args = parser.parse_args()

    attempt_dir = Path(args.attempt_dir)
    summary = json.loads((attempt_dir / "summary.json").read_text())
    metadata = json.loads((attempt_dir / "metadata.json").read_text())

    codex_command = " ".join(f"`{part}`" for part in metadata["codex_command"])
    scorer_command = " ".join(f"`{part}`" for part in metadata["scorer_command"])

    lines = [
        f"# Probe Report — {metadata['attempt']}",
        "",
        "## Commands",
        "",
        f"- codex probe command: {codex_command}",
        f"- scorer command: {scorer_command}",
        "",
        "## Per-variant results",
        "",
        "| Variant | n | mean P | stdev P | mean M | stdev M | min | max | scores | ceilings | integrity hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in summary["rows"]:
        ceilings = ", ".join(row["ceilings"]) if row["ceilings"] else "—"
        scores = ", ".join(str(score) for score in row["scores"])
        lines.append(
            f"| {row['variant']} | {row['n']} | {row['mean']:.2f} | {row['stdev']:.2f} | "
            f"{row['mean_M_training']:.4f} | {row['stdev_M_training']:.4f} | {row['min']} | {row['max']} | "
            f"[{scores}] | {ceilings} | {row['integrity_hits']} |"
        )

    acceptance = summary["acceptance"]
    lines.extend(
        [
            "",
            "## Layer A gates",
            "",
            f"- family_mean in [15, 25]: `{summary['family_mean']:.2f}` -> `{'PASS' if acceptance['family_mean_window'] else 'FAIL'}`",
            f"- max variant mean <= 40: `{summary['max_variant_mean']:.2f}` -> `{'PASS' if acceptance['max_variant_le_40'] else 'FAIL'}`",
            f"- at least one variant mean <= 10: `{summary['min_variant_mean']:.2f}` -> `{'PASS' if acceptance['at_least_one_variant_le_10'] else 'FAIL'}`",
            f"- monotonic V1>=V2>=V3>=V4>=V5 within +/-3: `{'PASS' if acceptance['monotonic_with_tolerance_3'] else 'FAIL'}`",
            "",
            "## Layer B probe metadata",
            "",
            f"- family mean M_training: `{summary['family_mean_M_training']:.4f}`",
            f"- current observed stdev M_training: `{summary['current_observed_stdev_M_training']:.4f}`",
            "",
            "## Artifacts",
            "",
            f"- attempt dir: `{attempt_dir}`",
            f"- runs jsonl: `{attempt_dir / 'probe_runs.jsonl'}`",
            f"- summary json: `{attempt_dir / 'summary.json'}`",
            f"- logs dir: `{attempt_dir / 'logs'}`",
            f"- saved artifacts dir: `{attempt_dir / 'artifacts'}`",
        ]
    )

    out = attempt_dir / f"{metadata['attempt']}_probe_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
