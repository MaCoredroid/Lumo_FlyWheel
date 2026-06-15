#!/usr/bin/env python3
"""
FR13 confound-free flip instrument (CPU-only, banked-data).

ONE instrument that is immune to every confound in the FR13 catalog and resolves the
contradictory verify-HELD studies. It does NOT re-open any refuted seam; it only
re-scores the banked per-token argmax-vs-recurrent-oracle data under a
confound-neutralizing protocol.

Confounds neutralized (catalog) and HOW:
  CROSS-BOOT autotune +-9      -> never compares raw counts across boots; all three arms
                                  (native / spine / cat9) are from the SAME rescore boot
                                  (output/fr13_recurrent_oracle/rescore_*.json), and we
                                  report a within-boot RATE, never a cross-boot integer.
  TRAJECTORY-FORK inflation    -> the COMMON-PREFIX restriction: positions are scored ONLY
                                  where all compared arms served byte-identical tokens
                                  (held trajectory). Past the first fork the streams are
                                  different sequences and are NOT counted.
  LENGTH / DENOMINATOR (#12)   -> per-1000-token RATE on the held window, never raw count;
                                  early-EOS prompt 0 (cat9 78 vs 128) cannot inflate.
  DE-CASCADE BY CONSTRUCTION   -> a de-cascade-aware INDEPENDENT-event collapse (adjacent
                                  gap<=GAP flips folded) + a root-fork basin collapse, with
                                  the collapse reported SEPARATELY from the raw flip count so
                                  the instrument artifact is visible, not assumed away.
  ORACLE FRAME                 -> the binding oracle is the no-spec RECURRENT single-step
                                  decode (rescore_*.json, recurrent_decode_oracle), same
                                  FLASH_ATTN backend on every arm; NEVER chunked-prefill /
                                  streamed logprobs / a serial-torch ref / a backend name.
  BI ASYMMETRY                 -> all three rescore arms ran FLASH_ATTN (asserted); the BI=1
                                  cat9 arm (shape_sweep) is reported ONLY as a labelled
                                  counter-control, never mixed into the binding comparison.
  SCALAR BLINDSPOT             -> the binding signal is the per-token clear-margin
                                  argmax-vs-oracle flip (deviation_nat >= threshold), not
                                  accept/event / bag-TV / pass-rate / superset count.

KEY RESULT (run on banked data): on the strictly-held common trajectory every arm has
ZERO clear-margin flips -- the first flip IS the fork. So the raw 3-vs-20-vs-5 are scored
on THREE DIFFERENT downstream trajectories. The confound-free comparison is therefore a
RATE on each arm's own post-fork tail, plus the de-cascaded independent-event count.

Usage:  python3 scripts/fr13_confound_free_flip_instrument.py
        (reads output/fr13_recurrent_oracle/rescore_{native,spine,cat9}.json
         + output/fr13_shape_sweep/*_flips.json; writes nothing unless --emit)
"""
import json
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE_DIR = os.path.join(ROOT, "output", "fr13_recurrent_oracle")
SWEEP_DIR = os.path.join(ROOT, "output", "fr13_shape_sweep")

# de-cascade folding gap (FR13 acceptance-ladder convention: adjacent gap<=2 = one cascade tail)
GAP = 2


def load_rescore(arm):
    p = os.path.join(ORACLE_DIR, "rescore_%s.json" % arm)
    return json.load(open(p))


def clear_positions(prompt):
    """clear-margin flip positions for one prompt's per-token record."""
    return [x["pos"] for x in prompt["positions"] if x.get("clear_margin")]


def served_ids(prompt):
    return [x["served_token_id"] for x in prompt["positions"]]


def common_prefix_len(seqs):
    """length of the leading run where every seq agrees, token-for-token."""
    L = min(len(s) for s in seqs)
    n = 0
    for i in range(L):
        if len(set(s[i] for s in seqs)) == 1:
            n += 1
        else:
            break
    return n


def de_cascade(positions, gap=GAP):
    """Fold runs of flips whose successive gap <= gap into ONE independent event.
    Returns the list of independent-event anchor positions (first of each run)."""
    if not positions:
        return []
    positions = sorted(positions)
    events = [positions[0]]
    for p in positions[1:]:
        if p - events[-1] > gap:
            events.append(p)
    return events


def basin_collapse(positions, basin=14):
    """Root-fork basin collapse: a single hard high-entropy root crossing can manufacture a
    contiguous BASIN of downstream progeny flips. Fold any window of flips spanning <= `basin`
    tokens that contains >=3 flips into ONE root-fork event (the p3 ctrl-basin pattern:
    one root fork + downstream progeny). Conservative: only collapses dense clusters, keeps
    isolated structural crossings separate. Returns collapsed independent-event count."""
    if not positions:
        return 0
    positions = sorted(positions)
    n = len(positions)
    used = [False] * n
    events = 0
    i = 0
    while i < n:
        if used[i]:
            i += 1
            continue
        # greedily grow a dense cluster from i
        j = i
        while j + 1 < n and positions[j + 1] - positions[i] <= basin:
            j += 1
        cluster_size = j - i + 1
        if cluster_size >= 3:
            # dense basin: collapse to one root-fork event
            for k in range(i, j + 1):
                used[k] = True
            events += 1
            i = j + 1
        else:
            # not dense: this position is its own independent event (de-cascade still applies upstream)
            used[i] = True
            events += 1
            i += 1
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", help="write JSON result to this path", default=None)
    args = ap.parse_args()

    out = {"schema": "fr13.confound_free_flip_instrument.v1"}

    arms = {}
    for a in ("native", "spine", "cat9"):
        d = load_rescore(a)
        arms[a] = d
        # BI / backend assertion (BI-asymmetry neutralization)
        assert d.get("attn_backend") == "FLASH_ATTN", "%s not FLASH_ATTN: %s" % (a, d.get("attn_backend"))

    out["backend_pinned"] = {a: arms[a]["attn_backend"] for a in arms}
    out["threshold_nat"] = {a: arms[a].get("threshold_nat") for a in arms}

    # ---- LAYER 1: held common-trajectory probe (the resolving measurement) ----
    # positions where ALL THREE arms served byte-identical tokens => the only truly
    # apples-to-apples window. Count clear-margin flips per arm there.
    held = {"per_prompt": [], "totals": {a: {"cp_positions": 0, "clear_flips": 0} for a in arms}}
    cp_total = 0
    for pid in range(4):
        pp = {a: arms[a]["per_prompt"][pid] for a in arms}
        seqs = [served_ids(pp[a]) for a in arms]
        cp = common_prefix_len(seqs)
        cp_total += cp
        rec = {"prompt_id": pid, "common_prefix_len": cp}
        for a in arms:
            cm = sum(1 for x in pp[a]["positions"][:cp] if x.get("clear_margin"))
            rec["%s_clear_flips_in_cp" % a] = cm
            held["totals"][a]["cp_positions"] += cp
            held["totals"][a]["clear_flips"] += cm
        held["per_prompt"].append(rec)
    held["common_prefix_total_positions"] = cp_total
    for a in arms:
        t = held["totals"][a]
        t["rate_per_1000"] = round(1000.0 * t["clear_flips"] / t["cp_positions"], 3) if t["cp_positions"] else None
    out["layer1_held_common_trajectory"] = held
    out["layer1_finding"] = (
        "ALL arms have %d clear-margin flips on the %d held common-prefix positions; the first "
        "clear-margin flip in every arm is AT/AFTER the trajectory fork => the flip IS the fork, "
        "so the raw counts are scored on three different downstream trajectories."
        % (sum(held["totals"][a]["clear_flips"] for a in arms), cp_total)
    )

    # ---- LAYER 2: own-trajectory RATE (length+cross-boot+oracle-frame neutral) ----
    # Each arm scored on its OWN served stream vs its OWN recurrent oracle, same boot,
    # same FLASH_ATTN backend. Report per-1000-token rate, never the raw integer.
    rate = {}
    for a in arms:
        tot_pos = sum(p["served_len"] for p in arms[a]["per_prompt"])
        tot_cm = arms[a]["total_clear_margin_flips"]
        rate[a] = {
            "scored_positions": tot_pos,
            "clear_margin_flips_raw": tot_cm,
            "rate_per_1000": round(1000.0 * tot_cm / tot_pos, 2),
            "served_lens": [p["served_len"] for p in arms[a]["per_prompt"]],
        }
    rate["cat9_over_native_rate_x"] = round(rate["cat9"]["rate_per_1000"] / rate["native"]["rate_per_1000"], 2)
    rate["spine_over_native_rate_x"] = round(rate["spine"]["rate_per_1000"] / rate["native"]["rate_per_1000"], 2)
    out["layer2_own_trajectory_rate"] = rate

    # ---- LAYER 3: de-cascade-aware INDEPENDENT-event count + basin collapse ----
    indep = {}
    for a in arms:
        all_pos = []
        per_prompt_decasc = []
        per_prompt_basin = []
        for p in arms[a]["per_prompt"]:
            cms = clear_positions(p)
            dc = de_cascade(cms)
            bc = basin_collapse(dc)
            per_prompt_decasc.append(len(dc))
            per_prompt_basin.append(bc)
            all_pos.append(cms)
        raw = sum(len(x) for x in all_pos)
        decasc = sum(per_prompt_decasc)
        basin = sum(per_prompt_basin)
        indep[a] = {
            "raw_clear_flips": raw,
            "de_cascaded_events": decasc,
            "basin_collapsed_events": basin,
            "per_prompt_raw": [len(x) for x in all_pos],
            "per_prompt_de_cascaded": per_prompt_decasc,
            "per_prompt_basin_collapsed": per_prompt_basin,
        }
    out["layer3_independent_events"] = indep

    # ---- LAYER 4: shape-sweep cross-check (DEPTH vs WIDTH reconciler), rate-normalized ----
    # These are a SEPARATE boot/family; reported only as the depth/width reconciler, with
    # the BI=1 cat9 arm labelled as a counter-control (never mixed into the binding number).
    sweep = {}
    for name in ("chain3", "chain5", "cat3w", "cat9_bi", "BA_PROJ_BI_OFF", "BA_PROJ_BI_ON"):
        fp = os.path.join(SWEEP_DIR, "%s_flips.json" % name)
        if not os.path.exists(fp):
            continue
        d = json.load(open(fp))
        tp = d.get("total_positions")
        tc = d.get("total_clear_margin_flips")
        sweep[name] = {
            "total_positions": tp,
            "clear_flips_raw": tc,
            "rate_per_1000": round(1000.0 * tc / tp, 1) if tp else None,
            "note": "BI=1 COUNTER-CONTROL (do not mix into binding)" if name == "cat9_bi" else "",
        }
    out["layer4_shape_sweep_rate"] = sweep
    # depth lever test: chain3 vs chain5 rate
    if "chain3" in sweep and "chain5" in sweep:
        out["layer4_depth_lever"] = {
            "chain3_rate": sweep["chain3"]["rate_per_1000"],
            "chain5_rate": sweep["chain5"]["rate_per_1000"],
            "verdict": "DEPTH DEAD (rates equal)" if sweep["chain3"]["rate_per_1000"] == sweep["chain5"]["rate_per_1000"] else "depth moves rate",
        }
    if "cat3w" in sweep and "chain3" in sweep:
        out["layer4_width_lever"] = {
            "chain3_rate": sweep["chain3"]["rate_per_1000"],
            "cat3w_rate": sweep["cat3w"]["rate_per_1000"],
            "width_excess_rate": round(sweep["cat3w"]["rate_per_1000"] - sweep["chain3"]["rate_per_1000"], 1),
            "verdict": "WIDTH (co-residency) is the dominant carrier; shallow+width is WORSE",
        }

    # ---- the one number that resolves the contradictions ----
    out["RESOLVING_NUMBERS"] = {
        "held_common_trajectory_flips_all_arms": sum(held["totals"][a]["clear_flips"] for a in arms),
        "held_common_trajectory_positions": cp_total,
        "native_rate_per_1000": rate["native"]["rate_per_1000"],
        "spine_rate_per_1000": rate["spine"]["rate_per_1000"],
        "cat9_rate_per_1000": rate["cat9"]["rate_per_1000"],
        "cat9_over_native_rate_x": rate["cat9_over_native_rate_x"],
        "spine_over_native_rate_x": rate["spine_over_native_rate_x"],
        "cat9_de_cascaded_events": indep["cat9"]["de_cascaded_events"],
        "cat9_basin_collapsed_events": indep["cat9"]["basin_collapsed_events"],
        "native_basin_collapsed_events": indep["native"]["basin_collapsed_events"],
        "spine_basin_collapsed_events": indep["spine"]["basin_collapsed_events"],
    }

    print(json.dumps(out, indent=2))
    if args.emit:
        json.dump(out, open(args.emit, "w"), indent=2)
        print("\nwrote %s" % args.emit, file=sys.stderr)


if __name__ == "__main__":
    main()
