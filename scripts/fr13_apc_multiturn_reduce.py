#!/usr/bin/env python3
# FR13 APC MULTI-TURN ORACLE GATE — reducer.
#
# Aligns the per-boot replays by turn (seq) and localizes, PER TURN, where cache-ON's
# greedy (temp=0 == argmax) output first diverges from each reference. Two reference
# axes, both honoring the standing rule (US vs no-spec oracle / cache-OFF, never a proxy):
#
#   ORACLE (no-spec recurrent decode) = absolute ground truth.
#   CFG (chunked-prefill + max_num_batched=1024, NO cache restore) = the CONFIG-MATCHED
#        control. CFG and ON share the identical APC serve config and differ ONLY by the
#        cache bundle (fp32-SSM + prefix-restore). In the rate gate CFG resolved 3/3 and
#        ON 0/3, so per-turn ON-vs-CFG divergence isolates the cache effect with config
#        held constant — the tightest available isolation (the user's point).
#   OFF (base tree, no chunked/no 1024) = the loosest floor (conflates config + cache).
#
# Headline when --cfg is provided: does ON diverge from CFG (config-matched) on more /
# earlier turns than CFG itself diverges from the oracle, and does that excess GROW with
# turn index (the compounding signal behind the rate-gate cache-ON 0/3 solve collapse)?
import argparse, json


def first_div(a, b):
    """char offset of first difference between two canon strings; None if identical."""
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    if i == n and len(a) == len(b):
        return None
    return i


def _earlier(test, floor):
    """True if `test` diverges strictly earlier than `floor` (None == infinity == lossless)."""
    if test is None:
        return False
    if floor is None:
        return True
    return test < floor


def load(p):
    d = json.load(open(p))
    return {r.get("seq", r.get("idx")): r for r in d.get("records", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--cfg", default=None, help="config-matched control (chunked+1024, no cache) — the tight isolation")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    orc, off, on = load(a.oracle), load(a.off), load(a.on)
    cfg = load(a.cfg) if a.cfg else None
    common = set(orc) & set(off) & set(on)
    if cfg is not None:
        common &= set(cfg)
    keys = sorted(common, key=lambda k: (k is None, k))
    ordinal = {k: i for i, k in enumerate(keys)}

    rows = []
    for k in keys:
        co = orc[k].get("canon"); cf = off[k].get("canon"); cn = on[k].get("canon")
        cc = cfg[k].get("canon") if cfg is not None else None
        d_off = first_div(co, cf)     # base floor vs oracle
        d_on = first_div(co, cn)      # ON vs oracle
        d_cfg = first_div(co, cc) if cfg is not None else None   # CFG vs oracle (config-matched floor)
        d_on_cfg = first_div(cc, cn) if cfg is not None else None  # ON vs CFG (cache isolation, config held equal)
        rows.append({
            "seq": k,
            "oracle_len": len(co) if co else None,
            "off_div_vs_oracle": d_off,
            "cfg_div_vs_oracle": d_cfg,
            "on_div_vs_oracle": d_on,
            "on_div_vs_cfg": d_on_cfg,
            "on_div_vs_off": first_div(cf, cn),
            "on_cached_tokens": on[k].get("cached_tokens"),
            "cfg_cached_tokens": cfg[k].get("cached_tokens") if cfg is not None else None,
            "on_worse_than_oracle_floor": _earlier(d_on, d_off),
            "on_worse_than_cfg": (d_on_cfg is not None) if cfg is not None else None,
        })

    n = len(rows)
    half = n // 2
    on_cache_engaged = sum(1 for r in rows if (r["on_cached_tokens"] or 0) > 0)

    def split(pred):
        sel = [r for r in rows if pred(r)]
        first = sum(1 for r in sel if ordinal[r["seq"]] < half)
        return len(sel), first, len(sel) - first

    off_loss, _, _ = split(lambda r: r["off_div_vs_oracle"] is not None)
    on_loss, _, _ = split(lambda r: r["on_div_vs_oracle"] is not None)
    on_worse_floor, owf_1, owf_2 = split(lambda r: r["on_worse_than_oracle_floor"])

    summary = {
        "n_turns": n,
        "on_cache_engaged_turns": on_cache_engaged,
        "off_diverged_from_oracle_turns(base_floor)": off_loss,
        "on_diverged_from_oracle_turns": on_loss,
        "on_worse_than_base_floor_turns": on_worse_floor,
        "on_worse_than_base_floor_first/second_half": f"{owf_1}/{owf_2}",
    }

    if cfg is not None:
        cfg_engaged = sum(1 for r in rows if (r["cfg_cached_tokens"] or 0) > 0)
        cfg_loss, cfgl_1, cfgl_2 = split(lambda r: r["cfg_div_vs_oracle"] is not None)
        on_v_cfg, ovc_1, ovc_2 = split(lambda r: r["on_div_vs_cfg"] is not None)
        summary.update({
            "cfg_cache_engaged_turns(should_be_0)": cfg_engaged,
            "cfg_diverged_from_oracle_turns(config_matched_floor)": cfg_loss,
            "cfg_floor_first/second_half": f"{cfgl_1}/{cfgl_2}",
            "ON_vs_CFG_diverged_turns(CACHE_ISOLATION)": on_v_cfg,
            "ON_vs_CFG_first/second_half": f"{ovc_1}/{ovc_2}",
        })
        # VERDICT keyed on the config-matched isolation
        if on_cache_engaged == 0:
            verdict = "INVALID: cache never engaged on ON (cached_tokens==0 every turn) — boot/keying bug"
        elif on_v_cfg == 0:
            verdict = ("CACHE-ON MULTI-TURN LOSSLESS vs config-matched CFG (greedy/BI/eager): ON byte-matches "
                       "the working config-only arm on every turn. The temp-0.6 rate-gate failure is NOT a "
                       "greedy argmax defect -> escalate to cuda-graph + temp-0.6 same-seed regime.")
        elif on_v_cfg > cfg_loss and ovc_2 >= ovc_1:
            verdict = (f"COMPOUNDING CACHE DRIFT LOCALIZED (config held equal): ON diverges from the working "
                       f"config-matched CFG on {on_v_cfg}/{n} turns ({ovc_1} early / {ovc_2} late) vs CFG's own "
                       f"{cfg_loss}/{n} oracle-floor turns. The cache bundle (fp32-SSM + prefix-restore) injects "
                       f"per-turn argmax drift that accumulates over the multi-turn solve. Next: bisect fp32-SSM "
                       f"vs prefix-restore.")
        else:
            verdict = (f"CACHE-ON DIVERGES FROM CFG on {on_v_cfg}/{n} turns ({ovc_1} early / {ovc_2} late); cache "
                       f"implicated with config held equal, compounding trend not yet monotone at this turn count.")
    else:
        if on_cache_engaged == 0:
            verdict = "INVALID: cache never engaged on ON (cached_tokens==0 every turn) — boot/keying bug"
        elif on_worse_floor == 0 and on_loss <= off_loss:
            verdict = ("CACHE-ON MULTI-TURN LOSSLESS vs base floor (greedy/BI/eager). Add the config-matched CFG "
                       "arm (--cfg) for the tight isolation, or escalate to cuda-graph + temp-0.6.")
        elif on_worse_floor and owf_2 > owf_1:
            verdict = (f"COMPOUNDING CACHE DRIFT vs base floor: ON worse on {on_worse_floor}/{n} turns, late-"
                       f"concentrated ({owf_1} early / {owf_2} late). NOTE: base floor conflates config+cache — "
                       f"rerun with --cfg to hold config equal.")
        else:
            verdict = f"CACHE-ON worse than base floor on {on_worse_floor}/{n} turns; add --cfg to isolate."

    summary["VERDICT"] = verdict
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)

    print("\n=== FR13 APC MULTI-TURN ORACLE GATE — VERDICT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    hdr = "seq | orc_len | off_div | cfg_div | on_div | ON-vs-CFG | on_cached"
    print(f"\n  per-turn ({hdr}):")
    for r in rows:
        flag = "  <== ON diverges from config-matched CFG" if r.get("on_worse_than_cfg") else ""
        print(f"    {str(r['seq']):>4} | {str(r['oracle_len']):>6} | {str(r['off_div_vs_oracle']):>7} "
              f"| {str(r['cfg_div_vs_oracle']):>7} | {str(r['on_div_vs_oracle']):>6} "
              f"| {str(r['on_div_vs_cfg']):>9} | {str(r['on_cached_tokens']):>7}{flag}")
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
