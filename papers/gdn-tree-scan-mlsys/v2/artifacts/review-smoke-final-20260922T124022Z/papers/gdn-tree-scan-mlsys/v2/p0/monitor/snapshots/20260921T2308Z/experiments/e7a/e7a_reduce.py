#!/usr/bin/env python3
"""E7a reducer: turn one run directory (result_*.json + manifest.json + telemetry/) into SUMMARY.md + summary.json.

Host-side, CPU only. Labels every row by provenance (historical sha256 / synthetic seed+regime) and never
counts files or layers as prefixes.

Validity rules (review 03):
  * any comparison with nonfinite_candidate > 0 OR nonfinite_reference > 0 renders as INVALID and is never
    summarized as an error value (finite-only statistics from compare() are shown only for fully finite rows);
  * timing attribution keys on OWNERSHIP: compute-process PIDs in the 1 Hz inventory are compared with the
    container's recorded host PID set (`telemetry/container_host_pid.txt`). Unknown/foreign PIDs => shared;
    missing PID record => UNVERIFIED. Absence of sampled foreign work is a sampled observation (1 Hz), not a guarantee;
  * telemetry coverage gaps (> 2.5 s between utilization samples) are reported;
  * the in-band probe-drift criterion (|after-before|/before <= 5 %) is enforced per timing block, not just printed;
  * timing rows carry a WORK label (out dtype, exports, allocations) — they are not apples-to-apples speedups.
"""
from __future__ import annotations

import argparse
import csv
import collections
import datetime as dt
import json
import math
from pathlib import Path

PROBE_DRIFT_MAX = 0.05
GAP_SECONDS = 2.5

WORK_NOTES = {
    "A_prod_scan_verify_out16": "bf16 out store; no per-node state export; preallocated",
    "A_prod_replay_commit_deepest": "writes depth+1 fp32 state rows to the bank; preallocated",
    "A_prod_replay_commit_zero_accept": "writes 1 fp32 state row (root); preallocated",
    "A_legacy_scan_with_state_export": "bf16 out + FULL per-node fp32 state export (n_pad x VH x DV x DK x 4 B); preallocated",
    "B_legacyWY_fp32closed_with_state_export": "bf16 out + FULL per-node fp32 state export; preallocated",
    "B_legacyWY_bf16bnd_with_state_export": "bf16 out (+bf16 boundary taps) + FULL per-node fp32 state export; preallocated",
    "native_pk_one_token(context)": "1 token; helper allocates mixed/a/b/out per call (NOT matched work)",
}


def work_note(k: str) -> str:
    if k in WORK_NOTES:
        return WORK_NOTES[k]
    if k.startswith("native_sg_chain"):
        return "T-token sequential native chain (bf16 out, fp32 per-token state store); wrapper allocates o/final_state per call (NOT matched work)"
    if "_verify_out16" in k:
        return "bf16 out store + U (fp32) + cum_g; preallocated (matched-output variant)"
    if "_verify" in k:
        return "fp32 out store + U (fp32) + cum_g; preallocated (fp32 output, NOT matched to A's bf16 store)"
    if "_commit_deepest" in k:
        return "writes 1 fp32 state row (dst bank); preallocated"
    if k.startswith("native_sg_chain"):
        return "T-token sequential chain; wrapper allocates o/final_state per call (NOT matched work)"
    return ""


def fmt(x, nd=3):
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if x == 0:
            return "0"
        return f"{x:.{nd}g}" if (abs(x) < 1e-3 or abs(x) >= 1e4) else f"{x:.{nd}f}"
    if x is None:
        return "—"
    return str(x)


def cmp_cell(d, keys=("max_abs", "max_rel_to_refmax", "ulp32_max_sig")) -> str:
    if not isinstance(d, dict):
        return "—"
    nf_c, nf_r = d.get("nonfinite_candidate", 0) or 0, d.get("nonfinite_reference", 0) or 0
    if nf_c or nf_r:
        return f"**INVALID** (nonfinite cand={nf_c} ref={nf_r}; finite pairs={d.get('n_finite', '?')}/{d.get('n', '?')})"
    parts = []
    for k in keys:
        if k in d and d[k] is not None:
            short = k.replace("max_rel_to_refmax", "rel").replace("ulp32_max_sig", "ulp").replace("ulp16_max_sig", "ulp16").replace("max_abs", "abs")
            parts.append(f"{short}={fmt(d[k])}")
    return " ".join(parts)


def invalid(d) -> bool:
    return isinstance(d, dict) and bool((d.get("nonfinite_candidate") or 0) or (d.get("nonfinite_reference") or 0))


def _parse_ts(s: str):
    for f in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s.strip(), f)
        except ValueError:
            pass
    return None


def attribution(tel_dir: Path) -> dict:
    out = {"available": tel_dir.is_dir()}
    if not tel_dir.is_dir():
        out["label"] = "ATTRIBUTION UNVERIFIED (no telemetry directory)"
        return out
    # ownership record
    ours: set[str] | None = None
    pidf = tel_dir / "container_host_pid.txt"
    if pidf.exists():
        ours = set()
        for line in pidf.read_text().splitlines():
            if line.startswith("pid="):
                ours.add(line.split("=", 1)[1].strip())
            if line.startswith("descendants="):
                ours.update(p.strip() for p in line.split("=", 1)[1].split(",") if p.strip())
        ours.discard("")
        out["container_host_pids"] = sorted(ours)
    # utilization + coverage
    util, stamps = [], []
    uf = tel_dir / "gpu_util_1hz.csv"
    if uf.exists():
        for r in csv.reader(open(uf)):
            if r and r[0].strip()[:2] == "20":
                t = _parse_ts(r[0])
                try:
                    u = int(r[1].strip().rstrip(" %"))
                except (ValueError, IndexError):
                    continue
                if t is not None:
                    util.append(u); stamps.append(t)
    gaps = []
    for a, b in zip(stamps, stamps[1:]):
        d = (b - a).total_seconds()
        if d > GAP_SECONDS:
            gaps.append({"from": a.isoformat(), "to": b.isoformat(), "seconds": d})
    out.update({"util_samples": len(util), "util_zero_frac": (sum(u == 0 for u in util) / len(util)) if util else None,
                "util_gt5_count": sum(u > 5 for u in util), "util_max": max(util) if util else None,
                "window_utc": [stamps[0].isoformat(), stamps[-1].isoformat()] if stamps else None,
                "coverage_gaps": gaps})
    # compute-process inventory by PID ownership
    per_pid = collections.Counter()
    names = collections.defaultdict(set)
    seconds = set()
    pf = tel_dir / "compute_procs_1hz.csv"
    if pf.exists():
        for r in csv.reader(open(pf)):
            if len(r) > 2:
                pid, name = r[1].strip(), r[2].strip()
                seconds.add(r[0].strip())
                if pid in ("none", ""):
                    continue
                per_pid[pid] += 1
                names[pid].add(name)
    out["inventory_seconds"] = len(seconds)
    out["compute_pids_seen"] = {pid: {"samples": n, "names": sorted(names[pid])} for pid, n in per_pid.items()}
    if ours is None:
        out["label"] = "ATTRIBUTION UNVERIFIED (container host PID not recorded; PIDs seen: %s)" % (sorted(per_pid) or "none")
        out["foreign_pids"] = None
    else:
        foreign = {pid: n for pid, n in per_pid.items() if pid not in ours}
        out["foreign_pids"] = foreign
        if foreign:
            out["label"] = "SHARED-DEVICE (foreign compute PIDs %s present; seconds=%s)" % (sorted(foreign), sum(foreign.values()))
        elif not per_pid:
            out["label"] = "ATTRIBUTION UNRESOLVED (no compute process sampled at all — inventory may have missed the container)"
        else:
            out["label"] = "NO FOREIGN COMPUTE PROCESS SAMPLED (1 Hz inventory; only container PID(s) %s seen; sampled observation, not a guarantee)" % sorted(per_pid)
    if gaps:
        out["label"] += f" — COVERAGE GAPS: {len(gaps)}"
    return out


def probe_check(tm: dict) -> dict:
    b, a = tm.get("probe_matmul4096_fp16_us_before"), tm.get("probe_matmul4096_fp16_us_after")
    if not (isinstance(b, (int, float)) and isinstance(a, (int, float)) and b > 0):
        return {"drift": None, "pass": False, "reason": "probe missing"}
    drift = abs(a - b) / b
    return {"drift": drift, "pass": drift <= PROBE_DRIFT_MAX, "before_us": b, "after_us": a, "criterion": f"<= {PROBE_DRIFT_MAX:.0%}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    man = json.load(open(rd / "manifest.json")) if (rd / "manifest.json").exists() else json.load(open(rd / "manifest_start.json"))
    man_start = json.load(open(rd / "manifest_start.json")) if (rd / "manifest_start.json").exists() else None
    src_match = None
    if man_start and "source_sha256" in man:
        src_match = man_start["source_sha256"] == man["source_sha256"]
    results = [json.load(open(p)) for p in sorted(rd.glob("result_*.json"))]
    per = [r for r in results if "stage1_verify" in r]
    b4 = [r for r in results if "stage1_verify" not in r]
    att = attribution(rd / "telemetry")
    n_invalid = 0

    L = []
    L.append(f"# E7a run summary — `{rd.name}`\n")
    L.append(f"Manifest status **{man.get('status')}**; start/end source hashes match: **{fmt(src_match)}**; image `{man.get('image_digest')}`; torch {man.get('torch')} / triton {man.get('triton')}; GPU {man.get('gpu')} cc {man.get('capability')}; driver `{(man.get('driver') or '')[:60]}`.")
    L.append(f"Production kernel flags: {json.dumps({k: v for k, v in man.get('flags', {}).items() if k != 'env_FR13_FR10_FR12'})}. Triton cubins hashed: {len(man.get('triton_cache_cubins_sha256', {}))}.")
    L.append("Source sha256: " + ", ".join(f"`{k}`={v[:12]}…" for k, v in man.get("source_sha256", {}).items()))
    L.append(f"\n**Timing attribution:** {att.get('label')} — window {att.get('window_utc')}, util samples {att.get('util_samples')}, zero-util fraction {fmt(att.get('util_zero_frac'))}, samples >5 %: {att.get('util_gt5_count')}, compute PIDs seen: {att.get('compute_pids_seen')}, container PIDs: {att.get('container_host_pids')}, coverage gaps: {len(att.get('coverage_gaps') or [])}.")
    L.append("\nAll mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled. Rows marked **INVALID** contain non-finite values in candidate or reference and carry no error value.\n")

    L.append("## Operand sets\n")
    L.append("| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        pv = r["provenance"]
        prov = f"HISTORICAL sha256 {pv.get('sha256', '')[:12]}… ({pv.get('layer_prefix')})" if not r["synthetic"] else f"SYNTHETIC seed={pv.get('seed')} regime={pv.get('regime')}"
        g = r["gates"]
        L.append(f"| {i} | {r['label']} | {prov} | `{r['parents']}` | {r['n']} | {r['depth']} | {fmt(g['g_min'])} | {fmt(g['cum_g_min'])} | {fmt(g['P_min'])} | {fmt(g['decay_ratio_min_visible'])} | {fmt(g['fp32_underflow_risk'])} |")

    L.append("\n## References vs fp64 oracle (same rounded operands)\n")
    L.append("| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk state vs ITS OWN oracle (beta bf16-rt + div/sqrt) | seam magnitude fp64 (oracle_pk vs oracle, state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state max_abs |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        R = r["references"]
        cells = [R["native_sg_out32_vs_oracle"], R["native_sg_state_vs_oracle"], R["native_pk_out32_vs_oracle"], R["native_pk_state_vs_oracle"],
                 R.get("native_pk_state_vs_oracle_pk(beta_bf16rt+divsqrt)"), R.get("oracle_pk_vs_oracle_state(seam_magnitude_fp64)")]
        n_invalid += sum(invalid(c) for c in cells)
        L.append(f"| {i} | {cmp_cell(cells[0])} | {cmp_cell(cells[1])} | {cmp_cell(cells[2])} | {cmp_cell(cells[3])} | {cmp_cell(cells[4])} | {cmp_cell(cells[5], ('max_abs', 'max_rel_to_refmax'))} | {fmt(R['native_pk_vs_native_sg_out16_exact_frac'])} | {fmt(R.get('native_sg_fp32io_vs_bf16io_state_maxabs(recompiled_specialization)'))} |")

    mechs = [("A_prod", "A_prod_out32_vs_oracle", "A_prod_out16_vs_native_sg16_exact_frac", None),
             ("B_legacyWY fp32-closed", "B_legacyWY_fp32closed_out32_vs_oracle", None, None),
             ("B_legacyWY bf16-bnd", "B_legacyWY_bf16bnd_out32_vs_oracle", None, None),
             ("B_fs[ieee]", "B_fs[ieee]_out32_vs_oracle", "B_fs[ieee]_out16_vs_native_sg16_exact_frac", "B_fs[ieee]_U_vs_oracle_u"),
             ("C_nm[ieee]", "C_nm[ieee]_out32_vs_oracle", "C_nm[ieee]_out16_vs_native_sg16_exact_frac", "C_nm[ieee]_U_vs_oracle_u"),
             ("B_fs[tf32]", "B_fs[tf32]_out32_vs_oracle", "B_fs[tf32]_out16_vs_native_sg16_exact_frac", "B_fs[tf32]_U_vs_oracle_u"),
             ("C_nm[tf32]", "C_nm[tf32]_out32_vs_oracle", "C_nm[tf32]_out16_vs_native_sg16_exact_frac", "C_nm[tf32]_U_vs_oracle_u")]
    L.append("\n## Stage 1 — verifier outputs and correction factors (vs fp64 oracle; out16 exact fraction vs native spec-update kernel)\n")
    L.append("| id | mechanism | out32 vs oracle | out16 exact vs native_sg | factors U vs oracle u |")
    L.append("|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        s1 = r["stage1_verify"]
        for name, k32, k16, kU in mechs:
            n_invalid += invalid(s1.get(k32)) + (invalid(s1.get(kU)) if kU else 0)
            L.append(f"| {i} | {name} | {cmp_cell(s1.get(k32))} | {fmt(s1.get(k16)) if k16 else '—'} | {cmp_cell(s1.get(kU)) if kU else '—'} |")
    L.append("\nProduction-scan identity checks (NUMERIC equality folds +0/-0; BITWISE is integer-view equality): " + "; ".join(
        f"id {i}: out32 vs native_sg numeric={fmt(r['stage1_verify']['A_prod_out32_vs_native_sg32'].get('exact_frac_all'))} bitwise={fmt(r['stage1_verify'].get('A_prod_out32_vs_native_sg32_bitwise_frac'), 6)}, out16 vs native_sg numeric={fmt(r['stage1_verify'].get('A_prod_out16_vs_native_sg16_exact_frac'), 6)} bitwise={fmt(r['stage1_verify'].get('A_prod_out16_vs_native_sg16_bitwise_frac'), 6)}, out16 bytes equal payload serving_out={fmt(r['stage1_verify'].get('A_prod_out16_bytes_equal_payload_serving_out'))}"
        for i, r in enumerate(per, 1)))
    L.append("\nSeam ablation for the one-token decode reference (native_pk state vs fp64 oracles applying ONE convention; max_rel): " + "; ".join(
        f"id {i}: beta-bf16-rt only={cmp_cell(r['references'].get('native_pk_state_vs_oracle_betart_only'), ('max_rel_to_refmax',))}, div-sqrt only={cmp_cell(r['references'].get('native_pk_state_vs_oracle_divsqrt_only'), ('max_rel_to_refmax',))}, both={cmp_cell(r['references'].get('native_pk_state_vs_oracle_pk(beta_bf16rt+divsqrt)'), ('max_rel_to_refmax',))}"
        for i, r in enumerate(per, 1) if 'native_pk_state_vs_oracle_betart_only' in r['references']))
    L.append("\n`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export path skips the root update (source-inspected); when/which binary changed that path is NOT established. Its outputs are bit-identical to the production scan.")

    L.append("\n## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)\n")
    L.append("| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        s2 = r["stage2_commit_identical_factors"]
        ks = ["B_fs[ieee]_compact_with_oracle_factors_vs_oracle", "B_fs[ieee]_compact_with_oracle_factors_vs_native_sg", "C_nm[ieee]_compact_with_oracle_factors_vs_oracle", "B_fs[tf32]_compact_with_oracle_factors_vs_oracle", "C_nm[tf32]_compact_with_oracle_factors_vs_oracle"]
        n_invalid += sum(invalid(s2.get(k)) for k in ks)
        L.append(f"| {i} | " + " | ".join(cmp_cell(s2.get(k)) for k in ks) + " |")

    L.append("\n## Stage 3 — own factors + own commit (all accepted prefixes incl. zero-accept; vs oracle / native_sg / native_pk)\n")
    L.append("| id | mechanism | vs oracle | vs native_sg (spec-update) | exact frac vs native_sg | vs native_pk (one-token decode) |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        s3 = r["stage3_own_factors_own_commit"]
        rows = [("A_prod replay", s3.get("A_prod_replay_vs_oracle"), s3.get("A_prod_replay_vs_native_sg_state"), s3.get("A_prod_replay_vs_native_pk_state")),
                ("B_legacyWY fp32-closed (materialized)", s3.get("B_legacyWY_fp32closed_state_as_commit_vs_oracle"), None, None)]
        for tag in ("B_fs[ieee]", "C_nm[ieee]", "B_fs[tf32]", "C_nm[tf32]"):
            rows.append((f"{tag} compact", s3.get(f"{tag}_compact_own_vs_oracle"), s3.get(f"{tag}_compact_own_vs_native_sg"), s3.get(f"{tag}_compact_own_vs_native_pk")))
        for name, a, b, c in rows:
            n_invalid += invalid(a) + invalid(b) + invalid(c)
            L.append(f"| {i} | {name} | {cmp_cell(a)} | {cmp_cell(b)} | {fmt(b.get('exact_frac_all')) if isinstance(b, dict) and not invalid(b) else '—'} | {cmp_cell(c)} |")
    L.append("\nReplay repeat-launch bitwise check (immutable h0 row): " + "; ".join(f"id {i}: {fmt(r['stage3_own_factors_own_commit'].get('A_prod_replay_repeat_bitwise_equal_all_paths'))}" for i, r in enumerate(per, 1)))

    L.append("\n## Committed-state max_abs vs oracle by accepted depth (finite rows only; any non-finite comparison in a row => n/a)\n")
    for i, r in enumerate(per, 1):
        bd = r["commit_maxabs_by_depth"]
        s3 = r["stage3_own_factors_own_commit"]
        keys = ["A_prod_replay", "B_fs[ieee]_own", "C_nm[ieee]_own", "B_fs[ieee]_oraclefactors", "native_sg", "native_pk", "B_fs[tf32]_own"]
        row_invalid = {"A_prod_replay": invalid(s3.get("A_prod_replay_vs_oracle")), "B_fs[ieee]_own": invalid(s3.get("B_fs[ieee]_compact_own_vs_oracle")),
                       "C_nm[ieee]_own": invalid(s3.get("C_nm[ieee]_compact_own_vs_oracle")), "B_fs[tf32]_own": invalid(s3.get("B_fs[tf32]_compact_own_vs_oracle")),
                       "B_fs[ieee]_oraclefactors": invalid(r["stage2_commit_identical_factors"].get("B_fs[ieee]_compact_with_oracle_factors_vs_oracle")),
                       "native_sg": invalid(r["references"].get("native_sg_state_vs_oracle")), "native_pk": invalid(r["references"].get("native_pk_state_vs_oracle"))}
        L.append(f"\nid {i} ({r['label']}):\n")
        L.append("| depth | " + " | ".join(keys) + " |")
        L.append("|---|" + "---|" * len(keys))
        for d in sorted(bd, key=int):
            L.append(f"| {d} | " + " | ".join(("n/a (nonfinite)" if row_invalid.get(k) else fmt(bd[d].get(k))) for k in keys) + " |")

    L.append("\n## Controls\n")
    L.append("| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        c = r["controls"]; t = r.get("neumann_truncation_control", {})
        L.append(f"| {i} | {fmt(t.get('U_vs_oracle'))} / {fmt(t.get('out_vs_oracle'))} | {fmt(c['B_fs_npad32_vs_npad16']['out_exact_frac'])} / {fmt(c['B_fs_npad32_vs_npad16']['out_max_abs'])} | {fmt(c['C_nm_npad32_vs_npad16']['out_exact_frac'])} / {fmt(c['C_nm_npad32_vs_npad16']['out_max_abs'])} | {fmt(c['B_fs_sibling_reverse']['out_exact_frac'])} / {fmt(c['B_fs_sibling_reverse']['U_exact_frac'])} | {fmt(c['C_nm_sibling_reverse']['out_exact_frac'])} / {fmt(c['C_nm_sibling_reverse']['U_exact_frac'])} | {fmt(c['A_prod_sibling_reverse']['out_exact_frac'])} |")

    L.append("\n## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)\n")
    if per:
        L.append("| quantity | " + " | ".join(f"id {i}" for i in range(1, len(per) + 1)) + " |")
        L.append("|---|" + "---|" * len(per))
        for k in per[0]["transient_memory_bytes"]:
            L.append(f"| {k} | " + " | ".join(f"{r['transient_memory_bytes'].get(k, 0):,}" for r in per) + " |")

    L.append(f"\n## Timing — {att.get('label')}\n")
    L.append("CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1. Rows do DIFFERENT work (see WORK column); this is NOT an apples-to-apples kernel speedup table.")
    probes = {i: probe_check(r["timing_us"]) for i, r in enumerate(per, 1) if "timing_us" in r}
    if per and "timing_us" in per[0]:
        keys = []
        for r in per:
            for k, v in r.get("timing_us", {}).items():
                if isinstance(v, dict) and k not in keys:
                    keys.append(k)
        L.append("\n| kernel | WORK | " + " | ".join(f"id {i}" for i in range(1, len(per) + 1)) + " |")
        L.append("|---|---|" + "---|" * len(per))
        for k in keys:
            cells = []
            for i, r in enumerate(per, 1):
                v = r.get("timing_us", {}).get(k)
                s = f"{v['sync_median_us']:.1f} / {v['pipelined_mean_us']:.1f}" if isinstance(v, dict) else "—"
                if i in probes and not probes[i]["pass"]:
                    s += " ⚠probe-drift"
                cells.append(s)
            L.append(f"| {k} | {work_note(k)} | " + " | ".join(cells) + " |")
        L.append("\nIn-band probe-drift criterion (4096² fp16 matmul before→after, pass iff drift ≤ 5 %): " + "; ".join(
            f"id {i}: {fmt(p.get('before_us'), 5)}→{fmt(p.get('after_us'), 5)} µs drift={fmt(p.get('drift'))} **{'PASS' if p['pass'] else 'FAIL'}** [{per[i-1]['timing_us'].get('timing_block_start_utc')}–{per[i-1]['timing_us'].get('timing_block_end_utc')}]"
            for i, p in probes.items()))
        L.append(f"\niters={man.get('args', {}).get('iters')}, warmup={man.get('args', {}).get('warmup')}. Not a serving throughput measurement.")

    if b4:
        L.append("\n## B4 arm (SYNTHETIC, 4 requests sharing layer params, batched factor kernels)\n")
        for r in b4:
            L.append(f"{r['label']}: n={r['n']} depth={r['depth']}")
            for tag, d in r["results"].items():
                if isinstance(d, dict) and "per_request" in d:
                    L.append(f"- {tag}: per-request out32/U/commit max_abs vs own oracle = " + "; ".join(f"req{e['request']}: {fmt(e['out32_vs_oracle'])}/{fmt(e['U_vs_oracle'])}/{fmt(e['commit_vs_oracle'])}(target {e['commit_target']})" for e in d["per_request"]) + (f"; B4-vs-B1 bitwise ALL FOUR requests (out/U/commit): {d['B4_vs_B1_bitwise_all_true']} {d['B4_vs_B1_bitwise_all_requests']}" if 'B4_vs_B1_bitwise_all_requests' in d else f"; B4-vs-B1 bitwise (request 2 ONLY): {d.get('B4_vs_B1_bitwise_request2')}") + (f"; timing verify {d['timing_us_B4_verify']['sync_median_us']:.1f}/{d['timing_us_B4_verify']['pipelined_mean_us']:.1f} µs, commit {d['timing_us_B4_commit']['sync_median_us']:.1f}/{d['timing_us_B4_commit']['pipelined_mean_us']:.1f} µs (batched fp32-out verify; 4 requests)" if "timing_us_B4_verify" in d else ""))
                elif isinstance(d, dict) and "sync_median_us" in d:
                    L.append(f"- {tag}: {d['sync_median_us']:.1f} / {d['pipelined_mean_us']:.1f} µs (4 sequential B1 launches, bf16 out)")
    L.append(f"\n## Validity\n\nINVALID (non-finite) comparison cells in this summary: **{n_invalid}**. Probe-drift failures: **{sum(not p['pass'] for p in probes.values())}**. Manifest status: **{man.get('status')}**; start/end source hashes match: **{fmt(src_match)}**.")

    (rd / "SUMMARY.md").write_text("\n".join(L) + "\n")
    json.dump({"run_dir": str(rd), "manifest_status": man.get("status"), "source_hashes_start_end_match": src_match, "attribution": att,
               "probe_checks": {str(k): v for k, v in probes.items()}, "n_invalid_cells": n_invalid, "n_operand_sets": len(per),
               "n_historical": sum(not r["synthetic"] for r in per), "n_synthetic": sum(r["synthetic"] for r in per),
               "b4_exercised": bool(b4), "independent_prefixes_established": 0}, open(rd / "summary.json", "w"), indent=1)
    print("\n".join(L[:5]))
    print(f"wrote {rd / 'SUMMARY.md'} (invalid cells={n_invalid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
