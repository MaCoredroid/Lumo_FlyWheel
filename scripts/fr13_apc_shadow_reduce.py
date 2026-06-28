#!/usr/bin/env python3
# FR13 APC SSM WRITE-SIDE SHADOW REDUCER (CPU, offline).
#
# Reads the write-side shadow log produced by the patcher's _fr13_apc_shadow_log
# (FR13_APC_SHADOW=1) and the proxy pair-dumps, and answers ONE question:
#   on an APC cache HIT, WHICH row class (spine / branch / zero-accept) does the
#   align RESTORE get wrong (i.e. restores the block-aligned row that does NOT
#   match the committed accepted-leaf row the committer wrote => STALE recurrent
#   SSM seed => the empty-patch carrier)?
#
# INPUTS (under $ARMDIR, default "."):
#   $ARMDIR/apc_shadow.jsonl            one record per gated SSM snapshot/restore
#                                       (see _fr13_apc_shadow_log fields)
#   $ARMDIR/proxy_pair_dumps/*.json     captured turns; response.usage.
#                                       input_tokens_details.cached_tokens > 0
#                                       proves a real cache hit happened
#
# GATE 1 (NON-VACUITY) runs FIRST: assert >=1 pair-dump with cached_tokens>0 AND
#   >=1 shadow record with is_cache_hit_row=True. Else {"non_vacuous": false} + exit 2.
# GATE 2: group cache-hit rows by row class (spine/branch/zero-accept) and by layer.
#   Two orthogonal lenses:
#     VALUE   : on FIRED rows (carrier "ssm", redirect_fired=True) the SNAP_FIX
#               redirect applied but still landed a stale row -> is_stale / index_equal
#               / max_abs / argmax. value_root = class with most is_stale fired rows.
#     COVERAGE: on NO-FIRE rows (carrier "ssm_nofire", redirect_fired=False) the
#               redirect did NOT apply (leaf absent or out of range) -> the STOCK
#               block-aligned (stale) row was served, invisible to the fired log.
#               coverage_root = class with the most redirect_fired=False rows.
# VERDICT: report BOTH value_root and coverage_root. The coverage_root is the
#   SNAP_FIX coverage-gap candidate ROOT. Diagnosed prediction = branch and/or
#   zero-accept is the value_root and/or coverage_root, spine clean.
#
# is_stale here = the int-view-bitwise stale bit recorded at write
# (restore_row != committed_leaf_row). The decisive write-side signal is the
# index inequality (restore_row_idx != committed_leaf_idx); the tensor compare
# (is_stale / max_abs / argmax_match) corroborates it.
import argparse
import glob
import json
import os
import sys


def _load_jsonl(path):
    recs = []
    if not os.path.isfile(path):
        return recs
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                # tolerate a torn final line
                pass
    return recs


def _cached_tokens(resp):
    # response.usage.input_tokens_details.cached_tokens (the responses-arm field
    # used by fr13_apc_shadow_gate.py / fr13_apc_multiturn_replay.py).
    return (
        ((resp.get("usage") or {}).get("input_tokens_details") or {})
        .get("cached_tokens", None)
    )


def _pair_dump_responses(path):
    """Yield the response object(s) from a pair-dump json, tolerant of schema:
    {"request":..., "response":...}  OR  {"on":{"response":...},"off":...}  OR
    a raw response (has "usage"/"output")."""
    try:
        d = json.load(open(path))
    except Exception:
        return
    if not isinstance(d, dict):
        return
    # direct response wrapper(s)
    for k in ("response", "on", "off", "cache_on", "cache_off"):
        v = d.get(k)
        if isinstance(v, dict):
            # nested {"response": {...}}
            if isinstance(v.get("response"), dict):
                yield v["response"]
            elif "usage" in v or "output" in v:
                yield v
    # raw response form
    if "usage" in d or "output" in d:
        yield d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--armdir",
        default=os.environ.get("ARMDIR", "."),
        help="arm dir holding apc_shadow.jsonl + proxy_pair_dumps/ (default $ARMDIR or .)",
    )
    ap.add_argument(
        "--shadow-log",
        default=None,
        help="override path to apc_shadow.jsonl (default $ARMDIR/apc_shadow.jsonl)",
    )
    ap.add_argument(
        "--max-abs-floor",
        type=float,
        default=float(os.environ.get("FR13_APC_SHADOW_FLOOR", "1e-3")),
        help="max_abs above this (or argmax flip) => CONFIDENT stale for the value corroboration",
    )
    args = ap.parse_args()

    armdir = args.armdir
    shadow_path = args.shadow_log or os.path.join(armdir, "apc_shadow.jsonl")
    pair_glob = os.path.join(armdir, "proxy_pair_dumps", "*.json")

    shadow = _load_jsonl(shadow_path)

    # -------- GATE 1: NON-VACUITY (real cache hit + cache-hit shadow record) -------- #
    n_pair_hit = 0
    n_pair_seen = 0
    for p in sorted(glob.glob(pair_glob)):
        for resp in _pair_dump_responses(p):
            n_pair_seen += 1
            ct = _cached_tokens(resp)
            if ct is not None and int(ct) > 0:
                n_pair_hit += 1
    n_shadow_hit = sum(
        1 for r in shadow if bool(r.get("is_cache_hit_row"))
    )

    if n_pair_hit < 1 or n_shadow_hit < 1:
        print(json.dumps({
            "non_vacuous": False,
            "reason": "no real cache hit and/or no cache-hit shadow record",
            "shadow_log": shadow_path,
            "shadow_records": len(shadow),
            "shadow_cache_hit_records": n_shadow_hit,
            "pair_dumps_seen": n_pair_seen,
            "pair_dumps_cached_tokens_gt0": n_pair_hit,
        }, indent=2))
        sys.exit(2)

    # -------- GATE 2: group cache-hit rows by class + layer -------- #
    def _class_of(r):
        if bool(r.get("is_zero_accept")):
            return "zero_accept"
        if bool(r.get("is_branch_row")):
            return "branch"
        if bool(r.get("is_spine_row")):
            return "spine"
        return "unknown"

    classes = {}
    by_layer = {}
    for r in shadow:
        if not bool(r.get("is_cache_hit_row")):
            continue
        cls = _class_of(r)
        # bitwise write-side stale bit (restore row != committed leaf row).
        is_stale = bool(r.get("is_stale"))
        # index inequality = the decisive write-side signal (does SNAP_FIX point
        # the restore at the committed-leaf row for this class?).
        idx_eq = r.get("index_equal")
        idx_stale = (idx_eq is False)
        # value-corroborated CONFIDENT stale (above floor OR argmax flip).
        ma = r.get("max_abs")
        am = r.get("argmax_match")
        confident = False
        try:
            confident = (
                (ma is not None and float(ma) == float(ma) and float(ma) > args.max_abs_floor)
                or (am is False)
            )
        except Exception:
            confident = False

        # COVERAGE: did the SNAP_FIX redirect fire for this cache-hit row?
        # redirect_fired=False (carrier "ssm_nofire") = the row was served the
        # STOCK block-aligned (stale) state -> a coverage-gap candidate ROOT.
        # Older records without the field default to fired=True (no-fire log is
        # a strict superset, so absence == fired).
        redirect_fired = bool(r.get("redirect_fired", True))
        leaf_present = r.get("leaf_present")

        c = classes.setdefault(cls, {
            "rows": 0, "stale": 0, "idx_stale": 0, "confident_stale": 0,
            "max_abs_max": 0.0, "argmax_flips": 0,
            "n_fired": 0, "n_nofire": 0, "n_nofire_leaf_absent": 0,
        })
        c["rows"] += 1
        if redirect_fired:
            c["n_fired"] += 1
        else:
            c["n_nofire"] += 1
            if leaf_present is False:
                c["n_nofire_leaf_absent"] += 1
        # is_stale / index / confident stats are meaningful only for FIRED rows
        # (no-fire rows have x_committed=None -> is_stale defaults True, no ref);
        # gate them on redirect_fired so the value-root reflects real compares.
        if redirect_fired:
            if is_stale:
                c["stale"] += 1
            if idx_stale:
                c["idx_stale"] += 1
            if confident:
                c["confident_stale"] += 1
            if am is False:
                c["argmax_flips"] += 1
        try:
            if ma is not None and float(ma) == float(ma):
                c["max_abs_max"] = max(c["max_abs_max"], float(ma))
        except Exception:
            pass

        lk = str(r.get("layer"))
        lc = by_layer.setdefault(lk, {})
        lcc = lc.setdefault(cls, {
            "rows": 0, "stale": 0, "idx_stale": 0, "confident_stale": 0,
            "n_fired": 0, "n_nofire": 0,
        })
        lcc["rows"] += 1
        if redirect_fired:
            lcc["n_fired"] += 1
            lcc["stale"] += int(is_stale)
            lcc["idx_stale"] += int(idx_stale)
            lcc["confident_stale"] += int(confident)
        else:
            lcc["n_nofire"] += 1

    # per-class summary. Value stats (stale/idx/confident) are over FIRED rows
    # (the rows with a faithful committed-leaf reference). Coverage stats
    # (n_fired/n_nofire) are over all cache-hit rows of the class.
    summary = {}
    coverage = {}
    for cls, c in classes.items():
        fired = max(c["n_fired"], 1)
        rows = max(c["rows"], 1)
        summary[cls] = {
            "rows": c["rows"],
            "n_fired": c["n_fired"],
            "n_nofire": c["n_nofire"],
            "stale": c["stale"],
            "stale_frac": c["stale"] / fired,
            "idx_stale": c["idx_stale"],
            "idx_stale_frac": c["idx_stale"] / fired,
            "confident_stale": c["confident_stale"],
            "confident_stale_frac": c["confident_stale"] / fired,
            "max_abs_max": c["max_abs_max"],
            "argmax_flips": c["argmax_flips"],
        }
        coverage[cls] = {
            "rows": c["rows"],
            "n_fired": c["n_fired"],
            "n_nofire": c["n_nofire"],
            "n_nofire_leaf_absent": c["n_nofire_leaf_absent"],
            # fraction of this class's cache-hit rows where SNAP_FIX did NOT fire
            # (served the stock block-aligned / stale row).
            "nofire_frac": c["n_nofire"] / rows,
        }

    # -------- VALUE verdict: class with most is_stale FIRED rows -------- #
    value_ranked = sorted(
        summary.items(),
        key=lambda kv: (kv[1]["confident_stale"], kv[1]["confident_stale_frac"],
                        kv[1]["idx_stale"], kv[1]["n_fired"]),
        reverse=True,
    )
    value_root = value_ranked[0][0] if value_ranked else None

    # -------- COVERAGE verdict: class with most redirect_fired=False rows -------- #
    # (served-stale SNAP_FIX coverage gap = the likely real root the fired-only
    # log was blind to). Rank by raw no-fire count, then no-fire fraction.
    cov_ranked = sorted(
        coverage.items(),
        key=lambda kv: (kv[1]["n_nofire"], kv[1]["nofire_frac"], kv[1]["rows"]),
        reverse=True,
    )
    coverage_root = cov_ranked[0][0] if (cov_ranked and cov_ranked[0][1]["n_nofire"] > 0) else None

    spine = summary.get("spine", {})
    diagnosed_match = (
        (value_root in ("branch", "zero_accept")
         or coverage_root in ("branch", "zero_accept"))
        and float(spine.get("confident_stale_frac", 0.0)) == 0.0
    )

    out = {
        "non_vacuous": True,
        "shadow_log": shadow_path,
        "shadow_records": len(shadow),
        "shadow_cache_hit_records": n_shadow_hit,
        "pair_dumps_seen": n_pair_seen,
        "pair_dumps_cached_tokens_gt0": n_pair_hit,
        "max_abs_floor": args.max_abs_floor,
        "by_class": summary,
        "coverage": coverage,
        "by_layer": by_layer,
        # value_root = where the FIRED redirect still lands a stale row;
        # coverage_root = where the redirect does NOT fire (served stale). The
        # coverage_root is the SNAP_FIX coverage-gap candidate the fired-only
        # log could not see.
        "value_root": value_root,
        "coverage_root": coverage_root,
        "value_ranking": [
            {"class": k, "confident_stale": v["confident_stale"],
             "confident_stale_frac": v["confident_stale_frac"],
             "n_fired": v["n_fired"]}
            for k, v in value_ranked
        ],
        "coverage_ranking": [
            {"class": k, "n_nofire": v["n_nofire"],
             "nofire_frac": v["nofire_frac"],
             "n_nofire_leaf_absent": v["n_nofire_leaf_absent"], "rows": v["rows"]}
            for k, v in cov_ranked
        ],
        "diagnosed_prediction_matches": diagnosed_match,
        "diagnosed_prediction": (
            "branch and/or zero_accept is the value_root (fired-but-stale) "
            "and/or the coverage_root (redirect did not fire -> served stale); "
            "spine clean"
        ),
    }
    print(json.dumps(out, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
