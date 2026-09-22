#!/usr/bin/env python3
"""E7a reducer: turn one run directory (result_*.json + manifest.json + telemetry/) into SUMMARY.md + summary.json.

Host-side, CPU only. Labels every row by provenance (historical sha256 / synthetic seed+regime) and never
counts files or layers as prefixes. Timing tables carry the contention attribution computed from the
run's own telemetry (1 Hz utilization + 1 Hz compute-process inventory + in-band matmul probes).
"""
from __future__ import annotations

import argparse
import csv
import collections
import json
from pathlib import Path


def fmt(x, nd=3):
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if x == 0:
            return "0"
        return f"{x:.{nd}g}" if (abs(x) < 1e-3 or abs(x) >= 1e4) else f"{x:.{nd}f}"
    if x is None:
        return "—"
    return str(x)


def cmp_cell(d: dict | None, keys=("max_abs", "max_rel_to_refmax", "ulp32_max_sig")) -> str:
    if not isinstance(d, dict):
        return "—"
    parts = []
    for k in keys:
        if k in d:
            parts.append(f"{k.replace('max_rel_to_refmax', 'rel').replace('ulp32_max_sig', 'ulp').replace('ulp16_max_sig', 'ulp16').replace('max_abs', 'abs')}={fmt(d[k])}")
    if d.get("nonfinite_candidate"):
        parts.append(f"NONFINITE={d['nonfinite_candidate']}")
    return " ".join(parts)


def attribution(tel_dir: Path) -> dict:
    out = {"available": tel_dir.is_dir()}
    if not tel_dir.is_dir():
        return out
    util = []
    for r in csv.reader(open(tel_dir / "gpu_util_1hz.csv")) if (tel_dir / "gpu_util_1hz.csv").exists() else []:
        if r and r[0].strip()[:2] == "20":
            try:
                util.append(int(r[1].strip().rstrip(" %")))
            except ValueError:
                pass
    procs = collections.defaultdict(set)
    foreign = collections.Counter()
    if (tel_dir / "compute_procs_1hz.csv").exists():
        for r in csv.reader(open(tel_dir / "compute_procs_1hz.csv")):
            if len(r) > 2:
                name = r[2].strip()
                procs[r[0].strip()].add(name)
                if name not in ("python3", "python", "none", "[No data]"):
                    foreign[name] += 1
    out.update({
        "util_samples": len(util), "util_zero_frac": (sum(u == 0 for u in util) / len(util)) if util else None,
        "util_gt5_count": sum(u > 5 for u in util), "util_max": max(util) if util else None,
        "inventory_seconds": len(procs), "foreign_compute_process_seconds": dict(foreign),
    })
    out["label"] = ("UNCONTENDED-WINDOW (no foreign compute process in the 1 Hz inventory; only our container)"
                    if procs and not foreign else "SHARED-DEVICE DIAGNOSTIC (foreign compute process present or inventory missing)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    man = json.load(open(rd / "manifest.json")) if (rd / "manifest.json").exists() else json.load(open(rd / "manifest_start.json"))
    results = [json.load(open(p)) for p in sorted(rd.glob("result_*.json"))]
    per = [r for r in results if "stage1_verify" in r]
    b4 = [r for r in results if "stage1_verify" not in r]
    att = attribution(rd / "telemetry")
    L = []
    L.append(f"# E7a run summary — `{rd.name}`\n")
    L.append(f"Manifest status **{man.get('status')}**; image `{man.get('image_digest')}`; torch {man.get('torch')} / triton {man.get('triton')}; GPU {man.get('gpu')} cc {man.get('capability')}; driver `{man.get('driver')}`.")
    L.append(f"Production kernel flags: {json.dumps({k: v for k, v in man.get('flags', {}).items() if k != 'env_FR13_FR10_FR12'})}. Triton cubins hashed: {len(man.get('triton_cache_cubins_sha256', {}))}.")
    L.append(f"Source sha256: " + ", ".join(f"`{k}`={v[:12]}…" for k, v in man.get("source_sha256", {}).items()))
    L.append(f"\nTiming attribution for this run: **{att.get('label')}** — util samples {att.get('util_samples')}, zero-util fraction {fmt(att.get('util_zero_frac'))}, samples >5 %: {att.get('util_gt5_count')}, foreign compute-process seconds: {att.get('foreign_compute_process_seconds')}.")
    L.append("\nAll mechanisms B/C are LOCAL reimplementations of the published mechanism families; nothing here measures the TreeWY or Bole authors' systems. Historical payloads are dependent historical operand sets (one topology, B1), not independent prefixes; synthetic rows are labeled.\n")

    # ---- provenance / gates
    L.append("## Operand sets\n")
    L.append("| id | label | provenance | topology | n | depth | g_min | cum_g_min | P_min | min visible decay ratio | fp32 underflow risk |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        pv = r["provenance"]
        prov = f"HISTORICAL sha256 {pv.get('sha256', '')[:12]}… ({pv.get('layer_prefix')})" if not r["synthetic"] else f"SYNTHETIC seed={pv.get('seed')} regime={pv.get('regime')}"
        g = r["gates"]
        L.append(f"| {i} | {r['label']} | {prov} | `{r['parents']}` | {r['n']} | {r['depth']} | {fmt(g['g_min'])} | {fmt(g['cum_g_min'])} | {fmt(g['P_min'])} | {fmt(g['decay_ratio_min_visible'])} | {fmt(g['fp32_underflow_risk'])} |")

    # ---- references
    L.append("\n## References vs fp64 oracle (same rounded operands)\n")
    L.append("| id | native_sg out32 | native_sg state | native_pk out32 vs oracle | native_pk state vs oracle | native_pk vs ITS OWN oracle (beta bf16-rt + div/sqrt) state | seam magnitude fp64 (oracle_pk vs oracle state) | pk vs sg out16 exact | sg fp32-io vs bf16-io state |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        R = r["references"]
        L.append(f"| {i} | {cmp_cell(R['native_sg_out32_vs_oracle'])} | {cmp_cell(R['native_sg_state_vs_oracle'])} | {cmp_cell(R['native_pk_out32_vs_oracle'])} | {cmp_cell(R['native_pk_state_vs_oracle'])} | {cmp_cell(R.get('native_pk_state_vs_oracle_pk(beta_bf16rt+divsqrt)'))} | {cmp_cell(R.get('oracle_pk_vs_oracle_state(seam_magnitude_fp64)'), ('max_abs', 'max_rel_to_refmax'))} | {fmt(R['native_pk_vs_native_sg_out16_exact_frac'])} | {fmt(R.get('native_sg_fp32io_vs_bf16io_state_maxabs(recompiled_specialization)'))} |")

    # ---- stage 1
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
            L.append(f"| {i} | {name} | {cmp_cell(s1.get(k32))} | {fmt(s1.get(k16)) if k16 else '—'} | {cmp_cell(s1.get(kU)) if kU else '—'} |")
    L.append("\nProduction-scan identity checks: " + "; ".join(
        f"id {i}: out32 bit-equal native_sg={fmt(r['stage1_verify']['A_prod_out32_vs_native_sg32'].get('exact_frac_all'))}, out16 bytes equal payload serving_out={fmt(r['stage1_verify'].get('A_prod_out16_bytes_equal_payload_serving_out'))}"
        for i, r in enumerate(per, 1)))
    L.append("\n`A_legacy_state*` rows are excluded: the June-8 vendored sequential scan's export skips the root update (see STATUS.md); its outputs are bit-identical to the production scan.")

    # ---- stage 2 / 3
    L.append("\n## Stage 2 — commit from IDENTICAL oracle factors (compact reconstruction kernel; vs oracle state / vs native_sg state)\n")
    L.append("| id | B_fs[ieee] vs oracle | vs native_sg | C_nm[ieee] vs oracle | B_fs[tf32] vs oracle | C_nm[tf32] vs oracle |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        s2 = r["stage2_commit_identical_factors"]
        L.append(f"| {i} | {cmp_cell(s2.get('B_fs[ieee]_compact_with_oracle_factors_vs_oracle'))} | {cmp_cell(s2.get('B_fs[ieee]_compact_with_oracle_factors_vs_native_sg'))} | {cmp_cell(s2.get('C_nm[ieee]_compact_with_oracle_factors_vs_oracle'))} | {cmp_cell(s2.get('B_fs[tf32]_compact_with_oracle_factors_vs_oracle'))} | {cmp_cell(s2.get('C_nm[tf32]_compact_with_oracle_factors_vs_oracle'))} |")
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
            L.append(f"| {i} | {name} | {cmp_cell(a)} | {cmp_cell(b)} | {fmt(b.get('exact_frac_all')) if isinstance(b, dict) else '—'} | {cmp_cell(c)} |")
    L.append("\nReplay repeat-launch bitwise check (immutable h0 row): " + "; ".join(f"id {i}: {fmt(r['stage3_own_factors_own_commit'].get('A_prod_replay_repeat_bitwise_equal_all_paths'))}" for i, r in enumerate(per, 1)))

    # ---- commit error by depth
    L.append("\n## Committed-state max_abs vs oracle by accepted depth\n")
    for i, r in enumerate(per, 1):
        bd = r["commit_maxabs_by_depth"]
        keys = ["A_prod_replay", "B_fs[ieee]_own", "C_nm[ieee]_own", "B_fs[ieee]_oraclefactors", "native_sg", "native_pk", "B_fs[tf32]_own"]
        L.append(f"\nid {i} ({r['label']}):\n")
        L.append("| depth | " + " | ".join(keys) + " |")
        L.append("|---|" + "---|" * len(keys))
        for d in sorted(bd, key=int):
            L.append(f"| {d} | " + " | ".join(fmt(bd[d].get(k)) for k in keys) + " |")

    # ---- controls
    L.append("\n## Controls\n")
    L.append("| id | Neumann d−1 terms (U err / out err) | B_fs N_PAD32 vs 16 (out exact / abs) | C_nm N_PAD32 vs 16 | B_fs sibling-reverse (out exact / U exact) | C_nm sibling-reverse | A_prod sibling-reverse out exact |")
    L.append("|---|---|---|---|---|---|---|")
    for i, r in enumerate(per, 1):
        c = r["controls"]; t = r.get("neumann_truncation_control", {})
        L.append(f"| {i} | {fmt(t.get('U_vs_oracle'))} / {fmt(t.get('out_vs_oracle'))} | {fmt(c['B_fs_npad32_vs_npad16']['out_exact_frac'])} / {fmt(c['B_fs_npad32_vs_npad16']['out_max_abs'])} | {fmt(c['C_nm_npad32_vs_npad16']['out_exact_frac'])} / {fmt(c['C_nm_npad32_vs_npad16']['out_max_abs'])} | {fmt(c['B_fs_sibling_reverse']['out_exact_frac'])} / {fmt(c['B_fs_sibling_reverse']['U_exact_frac'])} | {fmt(c['C_nm_sibling_reverse']['out_exact_frac'])} / {fmt(c['C_nm_sibling_reverse']['U_exact_frac'])} | {fmt(c['A_prod_sibling_reverse']['out_exact_frac'])} |")

    # ---- memory
    L.append("\n## Transient / scratch memory (bytes, analytic from shapes; per request, per layer)\n")
    if per:
        m = per[0]["transient_memory_bytes"]
        L.append("| quantity | bytes (id 1) |")
        L.append("|---|---|")
        for k, v in m.items():
            L.append(f"| {k} | {v:,} |")
        L.append("\n(Depth-dependent rows differ per operand set; see result JSONs.)")

    # ---- timing
    L.append(f"\n## Timing — {att.get('label')} — CUDA events, per-launch sync median / pipelined mean (µs), one layer, B1\n")
    if per and "timing_us" in per[0]:
        keys = [k for k, v in per[0]["timing_us"].items() if isinstance(v, dict)]
        L.append("| kernel | " + " | ".join(f"id {i}" for i in range(1, len(per) + 1)) + " |")
        L.append("|---|" + "---|" * len(per))
        for k in keys:
            cells = []
            for r in per:
                v = r.get("timing_us", {}).get(k)
                cells.append(f"{v['sync_median_us']:.1f} / {v['pipelined_mean_us']:.1f}" if isinstance(v, dict) else "—")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("\nIn-band probes (4096² fp16 matmul, µs before → after): " + "; ".join(
            f"id {i}: {fmt(r['timing_us'].get('probe_matmul4096_fp16_us_before'), 5)} → {fmt(r['timing_us'].get('probe_matmul4096_fp16_us_after'), 5)} [{r['timing_us'].get('timing_block_start_utc')}–{r['timing_us'].get('timing_block_end_utc')}]"
            for i, r in enumerate(per, 1) if "timing_us" in r))
        L.append(f"\niters={man.get('args', {}).get('iters')}, warmup={man.get('args', {}).get('warmup')}. Legacy kernels include a full per-node state export (50 MB) in their launch; production scan exports no state; replay/compact commits write the committed row(s). Not a serving throughput measurement.")

    # ---- B4
    if b4:
        L.append("\n## B4 arm (SYNTHETIC, 4 requests sharing layer params, batched factor kernels)\n")
        for r in b4:
            L.append(f"{r['label']}: n={r['n']} depth={r['depth']}")
            for tag, d in r["results"].items():
                if isinstance(d, dict) and "per_request" in d:
                    L.append(f"- {tag}: per-request out32/U/commit max_abs vs own oracle = " + "; ".join(f"req{e['request']}: {fmt(e['out32_vs_oracle'])}/{fmt(e['U_vs_oracle'])}/{fmt(e['commit_vs_oracle'])}(target {e['commit_target']})" for e in d["per_request"]) + f"; B4-vs-B1 bitwise (request 2): {d['B4_vs_B1_bitwise_request2']}" + (f"; timing verify {d['timing_us_B4_verify']['sync_median_us']:.1f}/{d['timing_us_B4_verify']['pipelined_mean_us']:.1f} µs, commit {d['timing_us_B4_commit']['sync_median_us']:.1f}/{d['timing_us_B4_commit']['pipelined_mean_us']:.1f} µs" if "timing_us_B4_verify" in d else ""))
                elif isinstance(d, dict) and "sync_median_us" in d:
                    L.append(f"- {tag}: {d['sync_median_us']:.1f} / {d['pipelined_mean_us']:.1f} µs")

    (rd / "SUMMARY.md").write_text("\n".join(L) + "\n")
    json.dump({"run_dir": str(rd), "manifest_status": man.get("status"), "attribution": att, "n_operand_sets": len(per),
               "n_historical": sum(not r["synthetic"] for r in per), "n_synthetic": sum(r["synthetic"] for r in per),
               "independent_prefixes_established": 0}, open(rd / "summary.json", "w"), indent=1)
    print("\n".join(L[:6]))
    print(f"wrote {rd / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
