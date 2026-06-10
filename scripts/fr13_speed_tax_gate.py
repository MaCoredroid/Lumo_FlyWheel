#!/usr/bin/env python3
"""FR13 speed-tax gate: paired measured per-forward reduce + tree-size sweep.

MEASUREMENT RULES (bestiary class 12, FR13_BUG_CLASS_PLAYBOOK.md):
- NEVER present TPS/accept division as a measured per-forward number (that
  hand-roll was retracted twice). This script REFUSES to derive a per-forward
  number from TPS/accept (HandRollGuardError) and emits UNAVAILABLE when the
  /metrics basis is missing.
- The MEASURED per-forward basis is
      request_decode_time_seconds_sum / spec_decode_num_drafts_total
  from /metrics scraped before/after a window (the wacoxe6i2 method).
  Validity: spec_draft_tokens/spec_drafts == num_spec_tokens proves
  drafts == forwards for the window.
- decode_seconds is a per-request SUM under concurrency: it is valid as a
  RATIO at matched shape (same B / max_tokens / samples / prompts), NOT as an
  absolute latency. Ratios are only emitted when the pairing checks pass
  (pinned prompts byte-identical, same seeds, same flag-state incl. BI on
  BOTH arms — class 9: flag-state+seed headers in every artifact).

Capture is NOT re-implemented here: arms are run dirs produced by the
canonical probe (scripts/fr10_quick_decode_tps_probe.py, which embeds the
before/after /metrics delta as `metric_delta`) plus optional raw
metrics_before/metrics_after Prometheus snapshots where those were saved
(e.g. output/fr13_acceptance_ladder/*/metrics_{before,after}.txt).

TOPOLOGY / N_PAD CAP (read from scripts/fr10_phase4_patch_vllm_tree_gdn.py):
- TREE env = python literal list of child-index path tuples
  (vLLM speculative_token_tree); the launcher sets
  num_speculative_tokens = len(TREE) = N draft nodes.
- The GDN tree verifier builds masks over n = N+1 nodes (root + drafts) and
  pads to n_pad = 1 << (n-1).bit_length(); it raises NotImplementedError for
  n_pad > 16. Hence N <= 15 draft nodes. A 16-node tree (n=17 -> n_pad=32)
  is REJECTED by the cap; the widest/deepest runnable shapes are N=15.

TRAFFIC MODEL constants (w78aq6xum, exact):
- GDN state row = 3.146 MB x 48 GDN layers
- native E5 = 7 row-touches/layer/req
- legacy tree = 3N + 2a + 1 rows (scratch N + publish 2N + remap 2a + h0 1;
  at N=9 that is 27 + 2a + 1)
- replay route = 2 + a rows (N-invariant — THE scaling claim the sweep tests)
- weights 27 GB @ 273 GB/s = 98.9 ms floor

Subcommands:
  reduce  paired reduce over existing arm run dirs (CPU only)
  sweep   emit the launch/probe command matrix for the tree-size sweep
          (commands are WRITTEN, never executed: NO GPU from this script)
  fit     fit tax-vs-N against linear-in-N (legacy) and N-invariant (replay)
          predictions; output is a labeled FIT, not a measured number
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Reused infra: the canonical probe's metric-name map (do NOT re-implement).
# ---------------------------------------------------------------------------


def _load_probe_module():
    path = REPO / "scripts" / "fr10_quick_decode_tps_probe.py"
    spec = importlib.util.spec_from_file_location("fr10_quick_decode_tps_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_PROBE = _load_probe_module()
METRIC_NAMES: dict[str, str] = dict(_PROBE.METRIC_NAMES)

MEASURED_BASIS = (
    "request_decode_time_seconds_sum / spec_decode_num_drafts_total "
    "(/metrics window delta; per-request SUM under concurrency -> valid as a "
    "RATIO at matched shape, NOT absolute latency)"
)

CLASS12_RULE = (
    "bestiary class 12 (FR13_BUG_CLASS_PLAYBOOK.md): never present TPS/accept "
    "division as a measured per-forward number; raw counters only"
)

# ---------------------------------------------------------------------------
# Hand-roll guard
# ---------------------------------------------------------------------------


class HandRollGuardError(RuntimeError):
    """Raised when a per-forward number would be derived from TPS/accept."""


# Any of these keys in a "metric delta" means the caller is feeding a probe
# SUMMARY (which contains TPS/accept rates) instead of the raw counter delta.
FORBIDDEN_RATE_KEYS = frozenset(
    {
        "warm_decode_tps",
        "per_request_decode_tps_mean",
        "per_request_decode_tps_median",
        "per_request_decode_tps_min",
        "per_request_decode_tps_max",
        "returned_tokens_per_wall_s",
        "returned_tokens_per_decode_s",
        "metrics_sum_decode_tps_concurrency_deflated",
        "accepted_per_draft_event",
        "accepted_per_draft_token",
        "tps",
        "accept_per_event",
    }
)


def measured_per_forward(
    metric_delta: dict[str, Any] | None,
    *,
    source: str,
    expected_num_spec_tokens: int | None = None,
) -> dict[str, Any]:
    """Compute the measured per-forward basis from a raw /metrics delta.

    Refuses (HandRollGuardError) if the input smells like a TPS/accept
    summary rather than raw counters. Returns status=UNAVAILABLE when the
    counters are missing — it NEVER falls back to TPS/accept.
    """
    if source not in {"metrics_files", "probe_metric_delta"}:
        raise HandRollGuardError(
            f"per-forward source must be a /metrics delta, got {source!r}; "
            + CLASS12_RULE
        )
    if metric_delta is None:
        return {"status": "UNAVAILABLE", "reason": "no /metrics delta", "source": source}
    forbidden = sorted(FORBIDDEN_RATE_KEYS & set(metric_delta))
    if forbidden:
        raise HandRollGuardError(
            "refusing to derive per-forward from a rate/summary input "
            f"(keys {forbidden}); " + CLASS12_RULE
        )
    decode_seconds = metric_delta.get("decode_seconds")
    spec_drafts = metric_delta.get("spec_drafts")
    spec_draft_tokens = metric_delta.get("spec_draft_tokens")
    if not decode_seconds or not spec_drafts or float(spec_drafts) <= 0:
        return {
            "status": "UNAVAILABLE",
            "reason": "decode_seconds and/or spec_drafts missing from /metrics delta",
            "source": source,
        }
    tokens_per_draft = (
        float(spec_draft_tokens) / float(spec_drafts) if spec_draft_tokens else None
    )
    drafts_equal_forwards = (
        tokens_per_draft is not None
        and abs(tokens_per_draft - round(tokens_per_draft)) < 1e-9
        and (
            expected_num_spec_tokens is None
            or round(tokens_per_draft) == int(expected_num_spec_tokens)
        )
    )
    return {
        "status": "MEASURED",
        "basis": MEASURED_BASIS,
        "source": source,
        "decode_seconds_sum": float(decode_seconds),
        "spec_drafts": float(spec_drafts),
        "spec_draft_tokens": float(spec_draft_tokens) if spec_draft_tokens else None,
        "seconds_per_forward_sum_basis": float(decode_seconds) / float(spec_drafts),
        "tokens_per_draft": tokens_per_draft,
        "expected_num_spec_tokens": expected_num_spec_tokens,
        "drafts_equal_forwards": bool(drafts_equal_forwards),
    }


# ---------------------------------------------------------------------------
# Prometheus snapshot parsing (raw metrics_before/metrics_after files)
# ---------------------------------------------------------------------------


def parse_prom_snapshot(path: Path) -> dict[str, float]:
    """Parse a saved /metrics snapshot using the probe's METRIC_NAMES map."""
    out = {value: 0.0 for value in METRIC_NAMES.values()}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = METRIC_NAMES.get(name)
        if key is None:
            continue
        try:
            out[key] += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return out


def snapshot_delta(before: Path, after: Path) -> dict[str, float]:
    b = parse_prom_snapshot(before)
    a = parse_prom_snapshot(after)
    return {key: float(a.get(key, 0.0) - b.get(key, 0.0)) for key in a}


# ---------------------------------------------------------------------------
# Topology: TREE env format, shape catalog, N_PAD cap
# ---------------------------------------------------------------------------

N_PAD_CAP = 16  # fr10_phase4_patch_vllm_tree_gdn.py raises NotImplementedError above this


def _chain(depth: int) -> list[tuple[int, ...]]:
    return [(0,) * d for d in range(1, depth + 1)]


def _caterpillar(spine_depth: int, alt_specs: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    """spine of `spine_depth` + alternates (depth, child_index) per spec."""
    paths = _chain(spine_depth)
    for depth, child in alt_specs:
        paths.append((0,) * (depth - 1) + (child,))
    return paths


SHAPE_CATALOG: dict[str, list[tuple[int, ...]]] = {
    # existing arms
    "chain5": _chain(5),
    "caterpillar9": _caterpillar(5, [(2, 1), (3, 1), (4, 1), (5, 1)]),
    # wider at the deployed depth-5 (separates width from depth)
    "caterpillar12_w3_d5": _caterpillar(
        5, [(2, 1), (3, 1), (4, 1), (5, 1), (2, 2), (3, 2), (4, 2)]
    ),
    # deeper: depth-6, 13 nodes
    "caterpillar13_d6": _caterpillar(
        6, [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (2, 2), (3, 2)]
    ),
    # max under the cap: N=15 (n=16 -> n_pad=16)
    "caterpillar15_w3_d6": _caterpillar(
        6,
        [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (2, 2), (3, 2), (4, 2), (5, 2)],
    ),
    # documentation entry: REJECTED by the n_pad cap (n=17 -> n_pad=32 > 16)
    "node16_REJECTED": _caterpillar(
        6,
        [
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (6, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (6, 2),
        ],
    ),
    # the native MTP-5 reference is not a TREE topology; modeled separately
}

NATIVE_TOPOLOGY = "native_mtp5"


def parse_tree_literal(text: str) -> list[tuple[int, ...]]:
    """Parse a TREE env value; tolerates a leading word like 'caterpillar '."""
    bracket = text.find("[")
    if bracket < 0:
        raise ValueError(f"no '[' in TREE literal: {text!r}")
    paths = ast.literal_eval(text[bracket:])
    return [tuple(int(x) for x in p) for p in paths]


def tree_literal(paths: list[tuple[int, ...]]) -> str:
    return "[" + ", ".join(
        "(" + ", ".join(str(x) for x in p) + ("," if len(p) == 1 else "") + ")"
        for p in paths
    ) + "]"


def topology_stats(paths: list[tuple[int, ...]]) -> dict[str, Any]:
    n_nodes = len(paths)
    n = n_nodes + 1  # verifier adds the root
    n_pad = 1 << (n - 1).bit_length()
    return {
        "num_draft_nodes": n_nodes,
        "depth": max((len(p) for p in paths), default=0),
        "verifier_n": n,
        "n_pad": n_pad,
        "allowed_by_n_pad_cap": n_pad <= N_PAD_CAP,
        "n_pad_cap": N_PAD_CAP,
    }


def validate_topology(paths: list[tuple[int, ...]]) -> None:
    """Well-formedness: parents exist, sibling child-indices contiguous."""
    path_set = set(paths)
    for p in paths:
        if len(p) > 1 and p[:-1] not in path_set:
            raise ValueError(f"node {p} has no parent node {p[:-1]}")
        if p[-1] > 0:
            sibling = p[:-1] + (p[-1] - 1,)
            if sibling not in path_set:
                raise ValueError(f"node {p} skips sibling child-index {sibling}")


def resolve_topology(spec: str) -> tuple[str, list[tuple[int, ...]] | None]:
    if spec == NATIVE_TOPOLOGY:
        return spec, None
    if spec in SHAPE_CATALOG:
        return spec, SHAPE_CATALOG[spec]
    paths = parse_tree_literal(spec)
    return "custom", paths


# ---------------------------------------------------------------------------
# Traffic model (w78aq6xum constants, exact)
# ---------------------------------------------------------------------------

STATE_ROW_MB = 3.146
GDN_LAYERS = 48
NATIVE_ROW_TOUCHES = 7.0  # native E5, per layer per req
WEIGHTS_GB = 27.0
BW_GBPS = 273.0
WEIGHTS_FLOOR_MS = 98.9


def predicted_state_rows(route: str, n_nodes: int | None, accepted_per_event: float | None) -> float | None:
    a = accepted_per_event
    if route == "native":
        return NATIVE_ROW_TOUCHES
    if a is None:
        return None
    if route == "legacy":
        if n_nodes is None:
            return None
        # scratch N + publish 2N + remap 2a + h0 1 (at N=9: 27 + 2a + 1)
        return 3.0 * n_nodes + 2.0 * a + 1.0
    if route == "replay":
        # N-invariant: 2 + a
        return 2.0 + a
    return None


def traffic_model(route: str, n_nodes: int | None, accepted_per_event: float | None) -> dict[str, Any]:
    rows = predicted_state_rows(route, n_nodes, accepted_per_event)
    if rows is None:
        return {
            "status": "UNAVAILABLE",
            "reason": "need route in {native,legacy,replay} (+ N and measured accept/event for tree routes)",
        }
    state_gb = rows * STATE_ROW_MB * 1e-3 * GDN_LAYERS
    total_gb = WEIGHTS_GB + state_gb
    return {
        "status": "MODEL",
        "label": "traffic-model PREDICTION (w78aq6xum constants), not a measurement",
        "route": route,
        "num_draft_nodes": n_nodes,
        "accepted_per_event_used": accepted_per_event,
        "state_rows_per_layer_per_fwd": rows,
        "state_gb_per_fwd": state_gb,
        "weights_gb_per_fwd": WEIGHTS_GB,
        "total_gb_per_fwd": total_gb,
        "bandwidth_gbps": BW_GBPS,
        "predicted_floor_ms_per_fwd": total_gb / BW_GBPS * 1000.0,
        "weights_only_floor_ms": WEIGHTS_FLOOR_MS,
    }


# ---------------------------------------------------------------------------
# Arm loading + headers (class 9: flag-state + seed headers in every artifact)
# ---------------------------------------------------------------------------

REQUIRED_HEADER_FIELDS = (
    "seed",
    "temperature",
    "top_p",
    "batch_size",
    "max_tokens",
    "samples_per_prompt",
    "mode",
    "route",
    "topology",
    "prompts_sha256",
    "batch_invariant",
)


def _find_probe_json(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*_probe.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one *_probe.json in {run_dir}, found "
            f"{[c.name for c in candidates]}"
        )
    return candidates[0]


def _extract_boot_flags(run_dir: Path, window_name: str) -> dict[str, Any] | None:
    """Pull the boot flag-state for this window from the campaign run_header.json."""
    header_path = run_dir.parent / "run_header.json"
    if not header_path.exists():
        return None
    try:
        header = json.loads(header_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    boots = header.get("boots")
    if not isinstance(boots, dict):
        return None
    for boot_name, boot in boots.items():
        if not isinstance(boot, dict):
            continue
        windows = boot.get("windows") or []
        if window_name in windows:
            flags = {
                k: v
                for k, v in boot.items()
                if isinstance(v, (int, float, str)) and k not in {"windows", "purpose"}
            }
            flags["_boot"] = boot_name
            flags["_campaign_header"] = str(header_path)
            return flags
    return None


def load_arm(
    name: str,
    run_dir: Path,
    *,
    metrics_before: Path | None = None,
    metrics_after: Path | None = None,
    route: str | None = None,
    topology_spec: str | None = None,
    label: str | None = None,
    mode: str | None = None,
    cli_flags: dict[str, str] | None = None,
) -> dict[str, Any]:
    probe_path = _find_probe_json(run_dir)
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if mode is None:
        if "summary" in probe:
            summary = probe["summary"]
        elif len(probe.get("modes", {})) == 1:
            summary = next(iter(probe["modes"].values()))
        else:
            raise ValueError(
                f"arm {name}: multi-mode probe {probe_path}; pass --mode {name}=<mode>"
            )
    else:
        summary = probe["modes"][mode]
    mode = summary.get("mode")

    window_name = run_dir.name
    boot_flags = _extract_boot_flags(run_dir, window_name)
    flags: dict[str, Any] = {}
    if boot_flags:
        flags.update(boot_flags)
    if cli_flags:
        flags.update(cli_flags)

    # route inference: explicit > FR13_REPLAY_ROUTE flag > naive_mtp=native
    if route is None:
        if "FR13_REPLAY_ROUTE" in flags:
            route = "replay" if str(flags["FR13_REPLAY_ROUTE"]) == "1" else "legacy"
        elif mode == "naive_mtp":
            route = "native"

    # topology: explicit > campaign-header TREE > native
    topo_paths: list[tuple[int, ...]] | None = None
    topo_name: str | None = None
    if topology_spec is not None:
        topo_name, topo_paths = resolve_topology(topology_spec)
    elif isinstance(flags.get("TREE"), str):
        topo_paths = parse_tree_literal(flags["TREE"])
        topo_name = "from_campaign_header_TREE"
    elif route == "native":
        topo_name = NATIVE_TOPOLOGY
    topo_stats = topology_stats(topo_paths) if topo_paths is not None else None
    expected_n = (
        topo_stats["num_draft_nodes"]
        if topo_stats is not None
        else (5 if topo_name == NATIVE_TOPOLOGY else None)
    )

    # --- measured per-forward basis (NEVER from TPS/accept) ---
    if metrics_before is not None and metrics_after is not None:
        delta = snapshot_delta(metrics_before, metrics_after)
        per_forward = measured_per_forward(
            delta, source="metrics_files", expected_num_spec_tokens=expected_n
        )
        per_forward["metrics_before"] = str(metrics_before)
        per_forward["metrics_after"] = str(metrics_after)
    elif isinstance(summary.get("metric_delta"), dict):
        per_forward = measured_per_forward(
            summary["metric_delta"],
            source="probe_metric_delta",
            expected_num_spec_tokens=expected_n,
        )
    else:
        per_forward = measured_per_forward(None, source="probe_metric_delta")

    # raw numbers straight from counters / probe (no division by accept here)
    accept_per_event = summary.get("accepted_per_draft_event")
    warm_tps = summary.get("warm_decode_tps")
    prompts = probe.get("prompts") or []
    prompts_sha = hashlib.sha256(
        json.dumps(prompts, sort_keys=True).encode("utf-8")
    ).hexdigest()

    bi = flags.get("BATCH_INVARIANT")
    header = {
        "arm": name,
        "run_dir": str(run_dir),
        "probe_json": str(probe_path),
        "seed": probe.get("seed"),
        "temperature": probe.get("temperature"),
        "top_p": probe.get("top_p"),
        "batch_size": probe.get("batch_size"),
        "max_tokens": probe.get("max_tokens"),
        "samples_per_prompt": probe.get("samples_per_prompt"),
        "mode": mode,
        "route": route,
        "topology": topo_name,
        "prompt_count": len(prompts),
        "prompts_sha256": prompts_sha,
        "batch_invariant": None if bi is None else str(bi),
        "flag_state": flags or None,
        "label": label,
    }
    missing = [
        field
        for field in REQUIRED_HEADER_FIELDS
        if header.get(field) in (None, "")
    ]
    header["header_complete"] = not missing
    header["header_missing"] = missing

    tm = traffic_model(
        route or "",
        topo_stats["num_draft_nodes"] if topo_stats else None,
        float(accept_per_event) if accept_per_event is not None else None,
    )

    return {
        "header": header,
        "topology_stats": topo_stats,
        "raw": {
            "accepted_per_draft_event": accept_per_event,
            "accepted_per_draft_token": summary.get("accepted_per_draft_token"),
            "warm_decode_tps": warm_tps,
            "per_request_decode_tps_mean": summary.get("per_request_decode_tps_mean"),
            "per_request_decode_tps_median": summary.get(
                "per_request_decode_tps_median"
            ),
            "metric_delta": summary.get("metric_delta"),
        },
        "per_forward": per_forward,
        "traffic_model": tm,
    }


# ---------------------------------------------------------------------------
# Pairing checks + ratio (the only way a per-forward COMPARISON is emitted)
# ---------------------------------------------------------------------------

PAIRING_FIELDS = (
    "prompts_sha256",
    "seed",
    "temperature",
    "top_p",
    "batch_size",
    "max_tokens",
    "samples_per_prompt",
)


def pairing_check(arm: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for field in PAIRING_FIELDS:
        av, bv = arm["header"].get(field), base["header"].get(field)
        if av != bv:
            failures.append(f"{field}: arm={av!r} baseline={bv!r}")
    abi, bbi = arm["header"].get("batch_invariant"), base["header"].get("batch_invariant")
    if abi is not None and bbi is not None:
        if str(abi) != str(bbi):
            failures.append(f"batch_invariant: arm={abi!r} baseline={bbi!r} (BI must match on BOTH arms)")
    else:
        failures.append("batch_invariant: unknown on one or both arms (class 9: flag-state header required)")
    return {"matched": not failures, "failures": failures}


def per_forward_ratio(arm: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    pf_a, pf_b = arm["per_forward"], base["per_forward"]
    pairing = pairing_check(arm, base)
    if pf_a.get("status") != "MEASURED" or pf_b.get("status") != "MEASURED":
        return {
            "status": "UNAVAILABLE",
            "reason": "per-forward basis UNAVAILABLE on arm and/or baseline (never derived from TPS/accept)",
            "pairing": pairing,
        }
    invalid = []
    if not pf_a.get("drafts_equal_forwards"):
        invalid.append("arm drafts!=forwards (spec_draft_tokens/spec_drafts not the expected integer)")
    if not pf_b.get("drafts_equal_forwards"):
        invalid.append("baseline drafts!=forwards")
    if not pairing["matched"]:
        invalid.append("pairing mismatch: " + "; ".join(pairing["failures"]))
    if invalid:
        return {"status": "REFUSED", "reasons": invalid, "pairing": pairing}
    return {
        "status": "MEASURED_RATIO",
        "label": "ratio of decode_seconds_sum/spec_drafts at matched shape (the only valid use of the sum basis)",
        "ratio_vs_baseline": pf_a["seconds_per_forward_sum_basis"]
        / pf_b["seconds_per_forward_sum_basis"],
        "arm_seconds_per_forward_sum_basis": pf_a["seconds_per_forward_sum_basis"],
        "baseline_seconds_per_forward_sum_basis": pf_b["seconds_per_forward_sum_basis"],
        "pairing": pairing,
    }


def ladder_row(name: str, arm: dict[str, Any], baseline_name: str, ratio: dict[str, Any] | None) -> str:
    h = arm["header"]
    ts = arm["topology_stats"]
    pf = arm["per_forward"]
    tm = arm["traffic_model"]
    raw = arm["raw"]
    topo = h.get("topology") or "?"
    if ts:
        topo += f"(N={ts['num_draft_nodes']},d={ts['depth']},pad={ts['n_pad']})"
    pf_txt = (
        f"{pf['seconds_per_forward_sum_basis']:.4f}s/fwd(sum-basis,{pf['source']},"
        f"drafts==fwds={'Y' if pf.get('drafts_equal_forwards') else 'N'})"
        if pf.get("status") == "MEASURED"
        else "UNAVAILABLE"
    )
    if ratio is None:
        ratio_txt = "baseline"
    elif ratio.get("status") == "MEASURED_RATIO":
        ratio_txt = f"{ratio['ratio_vs_baseline']:.3f}x"
    else:
        ratio_txt = f"n/a({ratio.get('status')})"
    tm_txt = (
        f"pred={tm['total_gb_per_fwd']:.2f}GB/fwd floor={tm['predicted_floor_ms_per_fwd']:.1f}ms"
        if tm.get("status") == "MODEL"
        else "pred=n/a"
    )
    acc = raw.get("accepted_per_draft_event")
    warm = raw.get("warm_decode_tps")
    parts = [
        "FR13-TAX",
        f"arm={name}",
        f"route={h.get('route') or '?'}",
        f"topo={topo}",
        f"accept/event={acc:.4f}" if acc is not None else "accept/event=n/a",
        f"warm_tps={warm:.3f}" if warm is not None else "warm_tps=n/a",
        f"per_fwd={pf_txt}",
        f"ratio_vs_{baseline_name}={ratio_txt}",
        tm_txt,
        f"BI={h.get('batch_invariant')}",
        f"seed={h.get('seed')}",
        f"temp={h.get('temperature')}",
    ]
    if not h.get("header_complete"):
        parts.append(f"HDR-INCOMPLETE(missing={','.join(h['header_missing'])})")
    if h.get("label"):
        parts.append(f"label={h['label']}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# reduce subcommand
# ---------------------------------------------------------------------------


def _parse_kv_list(values: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--{what} expects name=value, got {item!r}")
        key, value = item.split("=", 1)
        out[key] = value
    return out


def cmd_reduce(args: argparse.Namespace) -> int:
    arms_spec = _parse_kv_list(args.arm, "arm")
    if args.baseline not in arms_spec:
        raise SystemExit(f"--baseline {args.baseline!r} is not among --arm names")
    before = _parse_kv_list(args.metrics_before, "metrics-before")
    after = _parse_kv_list(args.metrics_after, "metrics-after")
    routes = _parse_kv_list(args.route, "route")
    topos = _parse_kv_list(args.topology, "topology")
    labels = _parse_kv_list(args.label, "label")
    modes = _parse_kv_list(args.mode, "mode")
    flags_raw = _parse_kv_list(args.flags, "flags")

    arms: dict[str, dict[str, Any]] = {}
    for name, run_dir in arms_spec.items():
        cli_flags = None
        if name in flags_raw:
            cli_flags = dict(
                pair.split("=", 1) for pair in flags_raw[name].split(",") if "=" in pair
            )
        arms[name] = load_arm(
            name,
            Path(run_dir),
            metrics_before=Path(before[name]) if name in before else None,
            metrics_after=Path(after[name]) if name in after else None,
            route=routes.get(name),
            topology_spec=topos.get(name),
            label=labels.get(name),
            mode=modes.get(name),
            cli_flags=cli_flags,
        )

    base = arms[args.baseline]
    rows: list[str] = []
    for name, arm in arms.items():
        ratio = None if name == args.baseline else per_forward_ratio(arm, base)
        arm["ratio_vs_baseline"] = ratio
        rows.append(ladder_row(name, arm, args.baseline, ratio))

    incomplete = [n for n, a in arms.items() if not a["header"]["header_complete"]]
    result = {
        "schema": "fr13.speed_tax_gate.reduce.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "measurement_rules": {
            "class12": CLASS12_RULE,
            "basis": MEASURED_BASIS,
            "validity": "spec_draft_tokens/spec_drafts == num_spec_tokens proves drafts==forwards",
            "pairing": "pinned prompts byte-identical, same seeds, same flag-state (BI BOTH arms)",
        },
        "baseline": args.baseline,
        "arms": arms,
        "ladder_rows": rows,
        "header_incomplete_arms": incomplete,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in rows:
        print(row)
    if incomplete and args.strict_headers:
        print(f"STRICT-HEADERS FAIL: incomplete headers on arms {incomplete}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------
# sweep subcommand (command matrix generation only — NO GPU work here)
# ---------------------------------------------------------------------------

PINNED_PROMPTS = "output/fr13_acceptance_ladder/prompts_swe4.json"
BATTERY = (
    "--batch-size 1 --samples-per-prompt 1 --max-tokens 128 "
    "--temperature 0.0 --top-p 1.0 --seed 1313 --warmup-samples 0 --wait-health 60"
)


def _probe_cmd(win: str, mode: str, expected_n: int, logs: str, tree: bool) -> str:
    extra = (
        " --require-tree-engagement"
        f" --tree-sampler-debug-log {logs}/tree_sampler_debug.jsonl"
        f" --tree-accept-log {logs}/tree_path_lcp.jsonl"
        f" --expected-draft-count {expected_n}"
        if tree
        else ""
    )
    return (
        f"python3 scripts/fr10_quick_decode_tps_probe.py "
        f"--prompts-file {PINNED_PROMPTS} --modes {mode} {BATTERY} "
        f"--request-metrics-out {win}/request_metrics.jsonl "
        f"--out {win}/{Path(win).name}_probe.json{extra}"
    )


def cmd_sweep(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.run_root.rstrip("/")
    port = args.port

    shapes: dict[str, Any] = {}
    matrix: list[dict[str, Any]] = []
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "# FR13 speed-tax sweep command matrix — generated by fr13_speed_tax_gate.py sweep.",
        "# SERIAL GPU discipline: one boot at a time; the launcher runs recover_host_memory;",
        "# docker rm -f the container + verify nvidia-smi/MemAvailable between boots.",
        "# Engagement (class 9): tree windows assert via --require-tree-engagement; verify the",
        "# boot log shows the requested TREE before trusting any number.",
        "# Depth>5 shapes: the MTP head is applied recursively, but CONFIRM drafter engagement",
        "# (expected-draft-count assert) on first boot of any new shape.",
        "set -euo pipefail",
        "echo 'This is a command MATRIX. Review + run sections manually (GPU stages serialized).'",
        "exit 0",
        "",
    ]

    for shape_name in args.shapes:
        if shape_name not in SHAPE_CATALOG:
            raise SystemExit(f"unknown shape {shape_name!r}; catalog: {sorted(SHAPE_CATALOG)}")
        paths = SHAPE_CATALOG[shape_name]
        validate_topology(paths)
        stats = topology_stats(paths)
        literal = tree_literal(paths)
        shapes[shape_name] = {"tree": literal, **stats}
        if not stats["allowed_by_n_pad_cap"]:
            commands.append(
                f"# SHAPE {shape_name}: REJECTED — N={stats['num_draft_nodes']} -> n="
                f"{stats['verifier_n']} -> n_pad={stats['n_pad']} > {N_PAD_CAP} "
                "(fr10_phase4_patch_vllm_tree_gdn.py NotImplementedError). Max runnable N=15."
            )
            commands.append("")
            continue
        n = stats["num_draft_nodes"]
        for route in ("legacy", "replay"):
            win = f"{run_root}/{shape_name}_{route}_greedy"
            logs = f"{run_root}/boot_{shape_name}_{route}_logs"
            env = {
                "TREE": literal,
                "BATCH_INVARIANT": "1",
                "FR13_BI_TREE_ATTN": "1",
                "FR10_METRICS": "1",
                "GPU_UTIL": "0.82",
                "FR13_RUN_DIR": logs.rsplit("/logs", 1)[0],
                "LOG_DIR": logs,
            }
            if route == "replay":
                env["FR13_REPLAY_ROUTE"] = "1"
            env_str = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            arm_cmds = [
                f"# --- {shape_name} / {route} (N={n}, depth={stats['depth']}, n_pad={stats['n_pad']}) ---",
            ]
            if route == "replay":
                arm_cmds.append(
                    "# REPLAY ROUTE: requires branch fr13-replay-route@9d4d22e3 (launcher FR13_REPLAY_ROUTE"
                )
                arm_cmds.append(
                    "# passthrough is NOT on main). LABEL: replay accept-bug confounded live"
                )
                arm_cmds.append(
                    "# (b2 accept 2.02->1.58, FR13_ACCEPT_ONLY_GATE4_FAIL_BIND) — fix before binding accept;"
                )
                arm_cmds.append(
                    "# the per-forward RATIO at matched shape is the quantity this sweep tests."
                )
            arm_cmds += [
                f"{env_str} bash scripts/fr13_launch_forked_fa2_tree_server.sh",
                f"mkdir -p {win}",
                f"curl -s http://127.0.0.1:{port}/metrics > {win}/metrics_before.txt",
                _probe_cmd(win, "tree_mtp", n, logs, tree=True),
                f"curl -s http://127.0.0.1:{port}/metrics > {win}/metrics_after.txt",
                "# teardown: docker rm -f fr13-forked-fa2-tree; verify nvidia-smi + MemAvailable",
                "",
            ]
            commands += arm_cmds
            matrix.append(
                {
                    "shape": shape_name,
                    "route": route,
                    "window": win,
                    "env": env,
                    "expected_num_spec_tokens": n,
                    "label": "replay accept-bug confounded live (fix before binding accept)"
                    if route == "replay"
                    else None,
                }
            )

    # native reference boot (once)
    native_win = f"{run_root}/native_mtp5_greedy"
    native_env = (
        "BATCH_INVARIANT=1 FR10_METRICS=1 ATTENTION_BACKEND=FLASH_ATTN "
        "FR10_ENABLE_TREE_GDN=0 FR10_DECODE_MODE_DEFAULT=naive_mtp GPU_UTIL=0.86 "
        "SPEC_CONFIG='{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":5}'"
    )
    commands += [
        "# --- native MTP-5 reference (E5-class, FLASH_ATTN, naive_mtp) ---",
        f"{native_env} bash scripts/fr10_launch_speed_server.sh",
        f"mkdir -p {native_win}",
        f"curl -s http://127.0.0.1:{port}/metrics > {native_win}/metrics_before.txt",
        _probe_cmd(native_win, "naive_mtp", 5, "", tree=False),
        f"curl -s http://127.0.0.1:{port}/metrics > {native_win}/metrics_after.txt",
        "",
        "# --- reduce + fit (CPU) ---",
        "# python3 scripts/fr13_speed_tax_gate.py reduce --baseline native_mtp5 \\",
        f"#   --arm native_mtp5={native_win} \\",
        f"#   --metrics-before native_mtp5={native_win}/metrics_before.txt \\",
        f"#   --metrics-after native_mtp5={native_win}/metrics_after.txt \\",
        "#   ... one --arm/--metrics-before/--metrics-after/--route/--topology per window ... \\",
        f"#   --out {run_root}/sweep_reduce.json",
        f"# python3 scripts/fr13_speed_tax_gate.py fit --reduce {run_root}/sweep_reduce.json "
        f"--out {run_root}/sweep_fit.json",
    ]

    matrix_doc = {
        "schema": "fr13.speed_tax_gate.sweep.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pad_cap": {
            "cap": N_PAD_CAP,
            "rule": "verifier pads n=N+1 nodes to n_pad=1<<(n-1).bit_length(); n_pad>16 raises -> max N=15 draft nodes",
            "source": "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
        },
        "predictions_under_test": {
            "legacy": "per-forward state traffic linear in N: rows = 3N + 2a + 1",
            "replay": "per-forward state traffic N-invariant: rows = 2 + a",
        },
        "battery": f"pinned {PINNED_PROMPTS}, B=1 spp=1 max_tokens=128 greedy 0.0/1.0 seed 1313, warmup 0",
        "shapes": shapes,
        "arms": matrix,
    }
    (out_dir / "sweep_matrix.json").write_text(
        json.dumps(matrix_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    script_path = out_dir / "sweep_commands.sh"
    script_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    print(f"sweep matrix: {out_dir / 'sweep_matrix.json'}")
    print(f"sweep commands (review, do not blind-run): {script_path}")
    for name, info in shapes.items():
        status = "OK" if info["allowed_by_n_pad_cap"] else "REJECTED(n_pad cap)"
        print(
            f"  {name}: N={info['num_draft_nodes']} depth={info['depth']} "
            f"n_pad={info['n_pad']} {status}"
        )
    return 0


# ---------------------------------------------------------------------------
# fit subcommand: tax-vs-N vs the linear-in-N / N-invariant predictions
# ---------------------------------------------------------------------------


def _ols_linear(xs: list[float], ys: list[float]) -> dict[str, Any]:
    n = len(xs)
    if n < 2:
        return {"status": "INSUFFICIENT_POINTS", "n": n}
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx == 0:
        return {"status": "DEGENERATE_X", "n": n}
    slope = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sxx
    intercept = ybar - slope * xbar
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in residuals)
    sst = sum((y - ybar) ** 2 for y in ys)
    out: dict[str, Any] = {
        "status": "FIT",
        "model": "linear_in_N",
        "n": n,
        "intercept_s": intercept,
        "slope_s_per_node": slope,
        "sse": sse,
        "r2": (1.0 - sse / sst) if sst > 0 else None,
        "residuals": residuals,
    }
    if n > 2:
        s2 = sse / (n - 2)
        se_slope = math.sqrt(s2 / sxx)
        se_intercept = math.sqrt(s2 * (1.0 / n + xbar**2 / sxx))
        out["slope_ci95_normal_approx"] = [
            slope - 1.96 * se_slope,
            slope + 1.96 * se_slope,
        ]
        out["intercept_ci95_normal_approx"] = [
            intercept - 1.96 * se_intercept,
            intercept + 1.96 * se_intercept,
        ]
        out["ci_label"] = "normal-approx 95% CI; small-n, treat as indicative"
    else:
        out["ci_label"] = "n==2: exact fit, no residual dof, no CI"
    return out


def _fit_constant(ys: list[float]) -> dict[str, Any]:
    n = len(ys)
    if n < 1:
        return {"status": "INSUFFICIENT_POINTS", "n": n}
    mean = sum(ys) / n
    residuals = [y - mean for y in ys]
    sse = sum(r * r for r in residuals)
    out: dict[str, Any] = {
        "status": "FIT",
        "model": "N_invariant_constant",
        "n": n,
        "constant_s": mean,
        "sse": sse,
        "residuals": residuals,
    }
    if n > 1:
        se = math.sqrt(sse / (n - 1) / n)
        out["constant_ci95_normal_approx"] = [mean - 1.96 * se, mean + 1.96 * se]
        out["ci_label"] = "normal-approx 95% CI; small-n, treat as indicative"
    else:
        out["ci_label"] = "n==1: single point, no CI"
    return out


def cmd_fit(args: argparse.Namespace) -> int:
    points: dict[str, list[dict[str, Any]]] = {}
    native_pf: list[float] = []
    for reduce_path in args.reduce:
        doc = json.loads(Path(reduce_path).read_text(encoding="utf-8"))
        for name, arm in doc.get("arms", {}).items():
            pf = arm.get("per_forward", {})
            if pf.get("status") != "MEASURED" or not pf.get("drafts_equal_forwards"):
                continue  # fit only on valid measured points (class 12)
            route = arm["header"].get("route")
            ts = arm.get("topology_stats")
            if route == "native":
                native_pf.append(pf["seconds_per_forward_sum_basis"])
                continue
            if route not in {"legacy", "replay"} or ts is None:
                continue
            points.setdefault(route, []).append(
                {
                    "arm": name,
                    "reduce_file": reduce_path,
                    "N": ts["num_draft_nodes"],
                    "seconds_per_forward_sum_basis": pf["seconds_per_forward_sum_basis"],
                    "accept_per_event": arm["raw"].get("accepted_per_draft_event"),
                    "label": arm["header"].get("label"),
                }
            )

    native_ref = sum(native_pf) / len(native_pf) if native_pf else None
    fits: dict[str, Any] = {}
    table: list[str] = [
        "tax-vs-N (FIT, model-based — not a measured per-forward number)",
        "route | arm | N | s/fwd(sum-basis) | tax vs native(s) | label",
    ]
    for route, pts in sorted(points.items()):
        pts.sort(key=lambda p: p["N"])
        xs = [float(p["N"]) for p in pts]
        ys = [float(p["seconds_per_forward_sum_basis"]) for p in pts]
        for p in pts:
            tax = (
                p["seconds_per_forward_sum_basis"] - native_ref
                if native_ref is not None
                else None
            )
            p["tax_vs_native_s"] = tax
            table.append(
                f"{route} | {p['arm']} | {p['N']} | "
                f"{p['seconds_per_forward_sum_basis']:.4f} | "
                + (f"{tax:.4f}" if tax is not None else "n/a")
                + f" | {p.get('label') or ''}"
            )
        linear = _ols_linear(xs, ys)
        const = _fit_constant(ys)
        preferred = None
        if linear.get("status") == "FIT" and const.get("status") == "FIT":
            n = len(ys)
            mse_lin = linear["sse"] / (n - 2) if n > 2 else None
            mse_const = const["sse"] / (n - 1) if n > 1 else None
            if mse_lin is not None and mse_const is not None:
                preferred = "linear_in_N" if mse_lin < mse_const else "N_invariant_constant"
            else:
                preferred = "insufficient points for dof-adjusted comparison"
        fits[route] = {
            "points": pts,
            "prediction_for_route": "linear_in_N (rows=3N+2a+1)"
            if route == "legacy"
            else "N_invariant (rows=2+a)",
            "linear_in_N_fit": linear,
            "N_invariant_fit": const,
            "preferred_model_by_dof_adjusted_mse": preferred,
        }

    result = {
        "schema": "fr13.speed_tax_gate.fit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": "FIT — model-based summary over measured per-forward sum-basis points; not itself a measurement",
        "measurement_rules": {"class12": CLASS12_RULE, "basis": MEASURED_BASIS},
        "native_reference_seconds_per_forward": native_ref,
        "native_reference_points": len(native_pf),
        "fits": fits,
        "table": table,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\n".join(table))
    for route, fit in fits.items():
        print(f"{route}: prediction={fit['prediction_for_route']} preferred={fit['preferred_model_by_dof_adjusted_mse']}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reduce = sub.add_parser("reduce", help="paired reduce over existing arm run dirs")
    p_reduce.add_argument("--arm", action="append", required=True, help="name=run_dir (repeat)")
    p_reduce.add_argument("--baseline", required=True, help="arm name used as ratio denominator")
    p_reduce.add_argument("--metrics-before", action="append", help="name=path to saved /metrics snapshot")
    p_reduce.add_argument("--metrics-after", action="append", help="name=path to saved /metrics snapshot")
    p_reduce.add_argument("--route", action="append", help="name=native|legacy|replay")
    p_reduce.add_argument("--topology", action="append", help="name=<catalog shape|TREE literal|native_mtp5>")
    p_reduce.add_argument("--label", action="append", help="name=free-text label (e.g. accept-bug confounded)")
    p_reduce.add_argument("--mode", action="append", help="name=probe mode for multi-mode probes")
    p_reduce.add_argument("--flags", action="append", help="name=K=V,K=V flag-state overrides")
    p_reduce.add_argument("--strict-headers", action="store_true", help="exit 2 on incomplete headers")
    p_reduce.add_argument("--out", required=True)
    p_reduce.set_defaults(func=cmd_reduce)

    p_sweep = sub.add_parser("sweep", help="emit launch/probe command matrix (no GPU work)")
    p_sweep.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "chain5",
            "caterpillar9",
            "caterpillar12_w3_d5",
            "caterpillar13_d6",
            "caterpillar15_w3_d6",
            "node16_REJECTED",
        ],
    )
    p_sweep.add_argument("--run-root", default="output/fr13_speed_tax_sweep")
    p_sweep.add_argument("--port", type=int, default=9950)
    p_sweep.add_argument("--out-dir", required=True)
    p_sweep.set_defaults(func=cmd_sweep)

    p_fit = sub.add_parser("fit", help="fit tax-vs-N vs linear-in-N / N-invariant predictions")
    p_fit.add_argument("--reduce", action="append", required=True, help="reduce output JSON (repeat)")
    p_fit.add_argument("--out", required=True)
    p_fit.set_defaults(func=cmd_fit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
