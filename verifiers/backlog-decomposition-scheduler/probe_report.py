#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

VARIANT_ORDER = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads(args.summary_json.read_text())
    gate = summary["gate"]
    variants = summary["variants"]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def fmt_check(name: str, ok: bool, detail: str) -> str:
        return f"- [{'PASS' if ok else 'FAIL'}] {name}: {detail}"

    with args.out.open("w") as f:
        f.write(f"probe_run_id={summary['probe_run_id']}\n")
        f.write(f"model={summary['model']} reasoning={summary['reasoning']} repeats={summary['repeats']}\n\n")
        f.write("Per-variant summary\n")
        f.write("===================\n\n")
        for variant in VARIANT_ORDER:
            row = variants[variant]
            ceilings = ", ".join(f"{name}x{count}" for name, count in sorted(row["ceiling_hits"].items())) or "-"
            f.write(
                f"{variant}: n={row['n']} mean={row['mean']:.2f} stdev={row['stdev']:.2f} "
                f"min={row['min']} max={row['max']} scores={row['scores']} raw={row['raw_scores']} "
                f"mean_M={row['mean_M_training']:.4f} stdev_M={row['stdev_M_training']:.4f} ceilings={ceilings}\n"
            )
        f.write("\nLayer A gate\n")
        f.write("============\n\n")
        f.write(f"family_mean={gate['family_mean']:.2f}\n")
        f.write(f"max_variant_mean={gate['max_variant_mean']:.2f}\n")
        f.write(f"min_variant_mean={gate['min_variant_mean']:.2f}\n")
        f.write(f"max_observed_stdev_M_training={gate['max_observed_stdev_M_training']:.4f}\n\n")
        f.write(fmt_check("family_mean in [15,25]", gate["family_mean_ok"], f"{gate['family_mean']:.2f}") + "\n")
        f.write(fmt_check("max variant mean <= 40", gate["max_variant_ok"], f"{gate['max_variant_mean']:.2f}") + "\n")
        f.write(fmt_check("at least one variant mean <= 10", gate["hard_variant_ok"], f"{gate['min_variant_mean']:.2f}") + "\n")
        f.write(fmt_check("monotonic V1>=V2>=V3>=V4>=V5 +/-3", gate["monotonic_ok"], "; ".join(gate["monotonic_breaks"]) or "ok") + "\n")
        f.write("\n")
        f.write(f"overall={'LAYER_A_PASS' if gate['all_pass'] else 'LAYER_A_FAIL_HARDEN_NEEDED'}\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
