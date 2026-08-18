#!/usr/bin/env python3
"""Reduce split-K probe runs into a Tier-B qualification credential.

Mark's pass-64 ruling created a second door: an arm that cannot be
byte-identical to the incumbent by construction may still serve, on a
credential that records what it IS rather than asserting what it is not.
This builds that credential, and it is deliberately a pure function of the
probe artifacts so it can be tested without a GPU.

WHAT IT WILL NOT DO, and why each refusal is here rather than in a comment:

* It will not reduce a probe that ran against a binary other than the pinned
  one. A credential authorises a KERNEL; a probe of a different kernel is
  evidence about something else.
* It will not invent a bound. The bounds are loaded from the pre-registered
  file and their digest is checked against the sidecar's pin, so the file
  cannot be edited to fit the result being reduced.
* It will not declare its own verdict. It emits measurements; the sidecar's
  validator recomputes every bound from them. The runner then re-validates the
  credential it just wrote, so a credential that the validator would reject
  never reaches disk as a PASS.
* It will not accept a second process that measured something else. The
  cross-process determinism claim is checked by comparing the per-case digests
  key by key, not by trusting two files that both say "passed".

Usage:
  fr14_reduce_splitk_tierb_credential.py --probe A.json --probe B.json \\
      --bounds <bounds.json> --source-commit <sha> --patch-source <path> \\
      --out credential.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SCALE = "captured"
BF16_EPS = 2.0 ** -8


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


def _sidecar():
    return _module(REPO / "scripts/fr13_qrow32_b1_pass_sidecar.py",
                   "fr14_tierb_sidecar")


def _contract():
    sys.path.insert(0, str(REPO / "scripts"))
    return _module(REPO / "scripts/fr13_fixed32_contract.py",
                   "fr13_fixed32_contract")


def _determinism_key(probe: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {
        f"{r['scale']}|{r['seq_len']}|{r['seed']}":
            (r["output_sha16"], r["lse_sha16"])
        for r in probe["determinism"]
    }


def _captured(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("scale") == SCALE]


def reduce_probes(
    probes: list[dict[str, Any]], *, arm: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """(determinism, measurements, probe descriptor) from >= 2 probe runs."""
    if len(probes) < 2:
        raise ValueError(
            "a tier-b credential needs at least two independent probe "
            "processes: cross-process determinism cannot be shown from one"
        )
    for probe in probes:
        if probe.get("schema") != "fr14.splitk_fa2.probe.v1":
            raise ValueError("probe schema drifted")
        if probe.get("candidate_arm") != arm:
            raise ValueError(
                f"probe measured {probe.get('candidate_arm')!r}, not {arm!r}"
            )

    shas = {p["so_sha256"] for p in probes}
    if len(shas) != 1:
        raise ValueError("probe processes did not measure the same binary")
    so_sha256 = shas.pop()

    tags = [p.get("process_tag") for p in probes]
    if len(set(tags)) != len(tags):
        raise ValueError(
            "probe processes carry duplicate tags; cross-process determinism "
            "must compare distinct runs"
        )

    keys = [_determinism_key(p) for p in probes]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError(
            "determinism digests differ ACROSS processes: the combine is not "
            "reproducible run to run and the hard gate has failed"
        )
    in_process = all(
        bool(r["bitwise_identical"]) for p in probes for r in p["determinism"]
    )
    reps = min(int(r["reps"]) for p in probes for r in p["determinism"])
    determinism = {
        "cases": len(keys[0]),
        "reps_per_case": reps,
        "processes": len(probes),
        "process_tags": tags,
        "all_cases_bitwise_identical": in_process,
        "cross_process_digests_identical": True,
        "digests": {k: {"output_sha16": v[0], "lse_sha16": v[1]}
                    for k, v in sorted(keys[0].items())},
    }

    primary = probes[0]
    rows = _captured(primary["characterization"])
    if not rows:
        raise ValueError(f"probe has no {SCALE}-scale characterization rows")
    summary = primary["characterization_summary"][SCALE]
    hist = summary["output_ulp_histogram"]
    total = sum(hist.values())
    within_2 = hist["0"] + hist["1"] + hist["2"]

    exact = _captured(primary["exact_reference"])
    if not exact:
        raise ValueError(f"probe has no {SCALE}-scale exact-reference rows")
    reference_arm = primary["reference_arm"]

    measurements = {
        # B1 -- the hard gate's two predicates, named exactly as the
        # pre-registered predicate string spells them.
        "all_cases_bitwise_identical": in_process,
        "cross_process_digests_identical": True,
        # B2
        "output_ulp_le_2_fraction": within_2 / total,
        # B3 -- scale-free: how many bf16 steps of the tensor's OWN maximum is
        # the worst disagreement, taken at its worst over cases.
        "output_max_abs_delta_in_bf16_eps_of_reference_max": max(
            r["output"]["max_abs_delta"]
            / (r["output"]["reference_max_abs"] * BF16_EPS)
            for r in rows
            if r["output"]["reference_max_abs"] > 0
        ),
        # B4, B5
        "lse_max_ulp": summary["lse_max_ulp"],
        "lse_max_abs_delta": summary["lse_max_abs_delta"],
        # B6 -- the comparison that makes a flip rate mean something.
        "argmax_flips_vs_exact_candidate": sum(
            r[arm]["argmax_flips_vs_exact"] for r in exact
        ),
        "argmax_flips_vs_exact_incumbent": sum(
            r[reference_arm]["argmax_flips_vs_exact"] for r in exact
        ),
        "argmax_rows_compared": sum(r[arm]["argmax_rows"] for r in exact),
        # B7, B8 -- worst case, not mean: a single case that got materially
        # worse is the one worth refusing on.
        "output_rms_vs_exact_ratio_worst": max(
            r[arm]["output_rms_error"] / r[reference_arm]["output_rms_error"]
            for r in exact
        ),
        "lse_rms_vs_exact_ratio_worst": max(
            r[arm]["lse_rms_error"] / r[reference_arm]["lse_rms_error"]
            for r in exact
        ),
        # B9
        "nonfinite_disagreements": sum(
            r["output"]["nonfinite_disagreements"]
            + r["lse"]["nonfinite_disagreements"]
            for r in rows
        ),
        # Recorded for the reader, not gated on.
        "output_ulp_histogram": hist,
        "output_ulp_le_1_fraction": (hist["0"] + hist["1"]) / total,
        "output_max_abs_delta": summary["output_max_abs_delta"],
        "output_max_ulp_same_sign_significant": summary[
            "output_max_ulp_same_sign_significant"
        ],
        "reference_arm": reference_arm,
    }

    seq_lens = sorted({r["seq_len"] for r in rows})
    probe_descriptor = {
        "operand_scale": SCALE,
        "seq_lens": seq_lens,
        "seeds": len({r["seed"] for r in rows}),
        "output_elements": total,
        "determinism_reps": reps,
        "determinism_processes": len(probes),
        "exact_seq_lens": sorted({r["seq_len"] for r in exact}),
        "exact_seeds": len({r["seed"] for r in exact}),
        "geometry": primary["geometry"],
        "projection": primary["projection"],
        "operand_scale_source": primary["operand_scales"][SCALE],
        "so_sha256": so_sha256,
    }
    return determinism, measurements, probe_descriptor


def build_credential(
    probes: list[dict[str, Any]],
    *,
    arm: str,
    source_commit: str,
    patch_source_sha256: str,
    bounds: dict[str, Any],
    bounds_sha256: str,
    sidecar,
    contract,
) -> dict[str, Any]:
    determinism, measurements, probe = reduce_probes(probes, arm=arm)
    if probe["so_sha256"] != contract.QROW32_B1_SPLITK_FA2_SHA256:
        raise ValueError(
            "the probe did not measure the pinned binary: "
            f"{probe['so_sha256']} != {contract.QROW32_B1_SPLITK_FA2_SHA256}"
        )
    pinned = sidecar.TIERB_ARM_IDENTITY[arm]
    payload: dict[str, Any] = {
        "schema": sidecar.TIERB_SCHEMA,
        "tier": "B",
        "arm": arm,
        "authority": (
            "Mark, FR14 pass 64: live-A/B serving on a Tier-B credential; "
            "promoted-default only after exact16 QC parity"
        ),
        "grants": "live-A/B serving only",
        "does_not_grant": bounds["scope"]["does_not_grant"],
        "identity": {
            "arm": arm,
            "so_sha256": contract.QROW32_B1_SPLITK_FA2_SHA256,
            "so_size": contract.QROW32_B1_SPLITK_FA2_SIZE,
            "source_closure_sha256": pinned["source_closure_sha256"],
            "fa2_head": pinned["fa2_head"],
            "sass_digest_sha256": pinned["sass_digest_sha256"],
            "baseline_sass_digest_sha256": pinned[
                "baseline_sass_digest_sha256"
            ],
            "source_commit": source_commit,
            "patch_source_sha256": patch_source_sha256,
            "bounds_sha256": bounds_sha256,
        },
        "selector": {
            "sentinel": pinned["sentinel"],
            "num_splits": pinned["num_splits"],
        },
        "determinism": determinism,
        "measurements": measurements,
        "probe": probe,
    }
    # The verdict is RECOMPUTED here from the pre-registered bounds and
    # recorded for the reader -- but the validator recomputes it again from
    # the measurements rather than reading this, so editing it changes
    # nothing.
    payload["bounds_evaluation"] = sidecar.evaluate_tierb_bounds(
        bounds, measurements
    )
    payload["probe_strength"] = sidecar.evaluate_tierb_probe_strength(
        bounds, probe
    )
    payload["credential_sha256"] = sidecar.tierb_credential_digest(payload)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="append", required=True,
                    help="probe result JSON; pass at least twice")
    ap.add_argument("--bounds", required=True, type=Path)
    ap.add_argument("--arm", default="gqa_pair_splitk")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--patch-source-sha256", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sidecar = _sidecar()
    contract = _contract()
    bounds = sidecar.load_tierb_bounds(args.bounds)
    probes = [json.loads(Path(p).read_text()) for p in args.probe]

    payload = build_credential(
        probes,
        arm=args.arm,
        source_commit=args.source_commit,
        patch_source_sha256=args.patch_source_sha256,
        bounds=bounds,
        bounds_sha256=sidecar.TIERB_BOUNDS_SHA256,
        sidecar=sidecar,
        contract=contract,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )

    # Re-validate what was just written, through the same door a serve would
    # use. A credential the validator would refuse must not survive this
    # script as though it had passed.
    verdict = sidecar.validate_tierb_credential(
        args.out,
        arm=args.arm,
        expected_candidate_sha256=contract.QROW32_B1_SPLITK_FA2_SHA256,
        expected_source_commit=args.source_commit,
        expected_patch_source_sha256=args.patch_source_sha256,
        bounds_path=args.bounds,
    )
    print(json.dumps({
        "credential": str(args.out),
        "credential_sha256": payload["credential_sha256"],
        "arm": args.arm,
        "bounds_passed": payload["bounds_evaluation"]["bounds_passed"],
        "probe_strength_passed": payload["probe_strength"][
            "probe_strength_passed"],
        "determinism": {
            k: payload["determinism"][k]
            for k in ("cases", "reps_per_case", "processes",
                      "all_cases_bitwise_identical",
                      "cross_process_digests_identical")
        },
        "validated": verdict["grants"],
    }, indent=2, sort_keys=True))
    for row in payload["bounds_evaluation"]["bounds"]:
        mark = "PASS" if row["passed"] else "FAIL"
        print(f"  {row['id']} {mark:4s} {row['name']:48s} "
              f"measured={row['measured']} bound={row['bound']}")
    return 0 if payload["bounds_evaluation"]["bounds_passed"] else 3


if __name__ == "__main__":
    sys.exit(main())
