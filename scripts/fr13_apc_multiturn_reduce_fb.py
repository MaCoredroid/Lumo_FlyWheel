#!/usr/bin/env python3
# FR13 APC MULTI-TURN FLOOR-BRACKETED reducer (5 arms).
#
# Removes the cross-boot autotune-fork confound that the 4-arm gate could not: each
# config is booted TWICE (on/on2, cfg/cfg2) so the same-config cross-boot first-divergence
# IS the per-turn autotune floor. A per-turn ON-vs-CFG divergence counts as a REAL cache
# signal only if it forks STRICTLY EARLIER than that autotune floor (so it cannot be the
# boot-to-boot Triton/autotune fork that even two identical-config boots show).
#
# Arms: oracle (no-spec ground truth), on/on2 (full APC, two boots), cfg/cfg2 (config-only
# = chunked+1024, NO cache, two boots). Config held equal between on and cfg except the
# cache bundle (fp32-SSM + prefix-restore). Compare target = US vs no-spec oracle / CFG,
# never a proxy (standing rule). Greedy temp0 == argmax; same trajectory, all arms.
#
# TWO metrics:
#  (1) char-offset first-divergence, floor-subtracted (rigorous for subtle turns).
#  (2) COLD-RESTART: the model loses the multi-turn conversation and re-introduces the task
#      ("the user is asking me to fix a bug...") where the oracle continues it. This is a
#      GROSS semantic failure, inherently immune to the autotune floor (a 22-char fork can't
#      manufacture a task restart). Counted per arm; if on AND on2 cold-restart on the same
#      turns but cfg/cfg2/oracle do not, the cache is the carrier and it is DETERMINISTIC.
import argparse, json, re

COLD = re.compile(r"user is asking me to fix|asking me to fix a bug|wants me to fix|"
                  r"provided a series of statements|mix of code, pseudocode|"
                  r"the user has provided|user is asking me to", re.I)


def first_div(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return None if (i == n and len(a) == len(b)) else i


def load(p):
    return {r.get("seq", r.get("idx")): r for r in json.load(open(p)).get("records", [])}


def is_cold(canon):
    return bool(canon) and bool(COLD.search(canon[:300]))


def _lt_floor(test, floor):
    """test forks strictly earlier than floor (None == infinity == no fork)."""
    if test is None:
        return False
    if floor is None:
        return True
    return test < floor


def main():
    ap = argparse.ArgumentParser()
    for a in ("oracle", "on", "on2", "cfg", "cfg2"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    O, N, N2, C, C2 = (load(getattr(a, k)) for k in ("oracle", "on", "on2", "cfg", "cfg2"))
    keys = sorted(set(O) & set(N) & set(N2) & set(C) & set(C2), key=lambda k: (k is None, k))
    ordinal = {k: i for i, k in enumerate(keys)}

    rows = []
    for k in keys:
        co, cn, cn2, cc, cc2 = O[k].get("canon"), N[k].get("canon"), N2[k].get("canon"), C[k].get("canon"), C2[k].get("canon")
        F_on = first_div(cn, cn2)       # ON same-config cross-boot autotune floor
        F_cfg = first_div(cc, cc2)      # CFG same-config cross-boot autotune floor
        # conservative autotune floor for an on-vs-cfg pair: earliest either config self-forks
        floor = min([x for x in (F_on, F_cfg) if x is not None], default=None)
        d_on_cfg = first_div(cc, cn)    # ON vs CFG (config equal except cache) + cross-boot
        d_on_orc = first_div(co, cn)
        d_cfg_orc = first_div(co, cc)
        rows.append({
            "seq": k, "on_cached": N[k].get("cached_tokens"), "cfg_cached": C[k].get("cached_tokens"),
            "F_on_floor": F_on, "F_cfg_floor": F_cfg, "autotune_floor": floor,
            "on_vs_cfg": d_on_cfg, "on_vs_oracle": d_on_orc, "cfg_vs_oracle": d_cfg_orc,
            # REAL cache signal: ON-vs-CFG forks earlier than the autotune floor
            "cache_signal": _lt_floor(d_on_cfg, floor),
            "cold_on": is_cold(cn), "cold_on2": is_cold(cn2), "cold_cfg": is_cold(cc),
            "cold_cfg2": is_cold(cc2), "cold_oracle": is_cold(co),
        })

    n = len(rows)
    half = n // 2
    def split(pred):
        sel = [r for r in rows if pred(r)]
        e = sum(1 for r in sel if ordinal[r["seq"]] < half)
        return len(sel), e, len(sel) - e

    sig, sig_e, sig_l = split(lambda r: r["cache_signal"])
    # cold-restart determinism: turns where BOTH on boots cold-restart but NO cfg/oracle arm does
    cold_cache_det = [r for r in rows if r["cold_on"] and r["cold_on2"]
                      and not (r["cold_cfg"] or r["cold_cfg2"] or r["cold_oracle"])]
    cold_on_any = [r for r in rows if r["cold_on"]]
    cold_cfg_any = [r for r in rows if r["cold_cfg"] or r["cold_cfg2"]]
    on_cache_engaged = sum(1 for r in rows if (r["on_cached"] or 0) > 0)
    cfg_clean = sum(1 for r in rows if (r["cfg_cached"] or 0) == 0)
    ccd_e = sum(1 for r in cold_cache_det if ordinal[r["seq"]] < half)

    if on_cache_engaged == 0:
        verdict = "INVALID: cache never engaged on ON (cached==0 every turn) — boot/keying bug"
    elif cold_cache_det and not cold_cfg_any:
        verdict = (f"CACHE IS THE CARRIER (deterministic, config-exonerated, confound-free): {len(cold_cache_det)}/{n} "
                   f"turns BOTH ON boots cold-restart (lose the conversation) while NEITHER CFG boot nor the oracle does "
                   f"({ccd_e} early / {len(cold_cache_det)-ccd_e} late). Same config, two boots => not autotune. The cache "
                   f"bundle (fp32-SSM + prefix-restore) corrupts the accumulated multi-turn prefix. Next: bisect fp32-SSM "
                   f"(MAMBA_SSM_CACHE_DTYPE=bfloat16) vs prefix-restore (SNAP_FIX/committed-leaf write path).")
    elif cold_cfg_any:
        verdict = (f"CONFIG IMPLICATED: CFG (no cache) ALSO cold-restarts on {len(cold_cfg_any)}/{n} turns => the chunked+1024 "
                   f"config (not the cache) carries some context loss. Re-examine config arm. on-cold={len(cold_on_any)} cfg-cold={len(cold_cfg_any)}.")
    elif sig > 0:
        verdict = (f"CACHE SIGNAL (char-offset, floor-subtracted): ON forks from CFG earlier than the autotune floor on "
                   f"{sig}/{n} turns ({sig_e} early / {sig_l} late) with no gross cold-restart. Subtle per-turn cache drift; "
                   f"compounding {'late-weighted' if sig_l>=sig_e else 'early-weighted'}.")
    else:
        verdict = ("CACHE-ON LOSSLESS within the autotune floor (no cold-restart, no floor-beating char divergence): the "
                   "greedy/eager/temp0 path is clean => the deploy 0/3 is regime-specific (cuda-graph/temp0.6) => escalate.")

    summary = {
        "n_turns": n, "on_cache_engaged_turns": on_cache_engaged, "cfg_clean_turns(want=n)": cfg_clean,
        "cache_signal_turns(floor_subtracted)": f"{sig} ({sig_e} early/{sig_l} late)",
        "cold_restart_ON_turns": len(cold_on_any),
        "cold_restart_BOTH_ON_boots_only(cache_det)": f"{len(cold_cache_det)} ({ccd_e} early/{len(cold_cache_det)-ccd_e} late)",
        "cold_restart_CFG_turns(should_be_0)": len(cold_cfg_any),
        "cold_restart_oracle_turns(should_be_0)": sum(1 for r in rows if r["cold_oracle"]),
        "VERDICT": verdict,
    }
    json.dump({"summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    print("\n=== FR13 APC MULTI-TURN FLOOR-BRACKETED VERDICT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n  per-turn (seq|on_cached|F_on|F_cfg|floor|on_vs_cfg|signal|cold:on/on2/cfg/cfg2/orc):")
    for r in rows:
        cold = f"{int(r['cold_on'])}/{int(r['cold_on2'])}/{int(r['cold_cfg'])}/{int(r['cold_cfg2'])}/{int(r['cold_oracle'])}"
        flag = " <== CACHE" if r["cache_signal"] else (" <== COLD(cache)" if (r["cold_on"] and r["cold_on2"] and not r["cold_cfg"]) else "")
        print(f"    {str(r['seq']):>4} | {str(r['on_cached']):>6} | {str(r['F_on_floor']):>5} | {str(r['F_cfg_floor']):>5} "
              f"| {str(r['autotune_floor']):>5} | {str(r['on_vs_cfg']):>7} | {str(r['cache_signal']):>5} | {cold}{flag}")
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
