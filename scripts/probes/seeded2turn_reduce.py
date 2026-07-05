#!/usr/bin/env python3
"""seeded2turn_reduce.py — reduce the FR13 2-turn seeded paired-stream probe.

Reads the per-arm turn{1,2}_<seed>.json completions + hits.jsonl + arm_meta.json
produced by seeded2turn_run.sh and answers three questions at TOKEN level (free,
no capture), then optionally localizes the carrier at STATE level (pass-2 dumps).

PRIMARY (token level)
  (1) COLD carrier  — cross-boot, FLOOR-bracketed:
        per seed, first-divergent token index of a cache arm's TURN-1 stream vs the
        cache-OFF reference (A) TURN-1 stream. Subtract the cross-boot autotune
        FLOOR measured by A vs A' (cache-OFF bootA vs bootB). Only forks EARLIER
        than the floor are a real above-floor cache-config divergence (§28).
  (2) RESTORE carrier — SAME-BOOT, confound-FREE:
        within each cache-ON boot, first-divergent token index of TURN-1(miss) vs
        TURN-2(hit) at the SAME seed. Identical input, identical kernels/layout;
        the only difference is recompute-vs-restore => a fork is a pure restore-
        losslessness failure with ZERO cross-boot confound (item 1d). In resend
        mode this is exact (turn-2 re-sends turn-1's prompt).
  (3) REFOLD value — does arm C's restore fork (readout 2) come LATER / vanish vs
        arm B's, AND did refold actually execute (redirect_used>0, from meta)?
  + route-choice distribution per arm x turn (delegate / read_file / NO_TOOL).

SECONDARY (state level, pass-2)  --state-dir <arm>/logs/decode_gdn
  Streams the per-decode-step GDN-state dumps (FR13_DECODE_GDN_CAPTURE) one file at
  a time (map_location=cpu, freed immediately) and reports, per (layer, component in
  conv_state -> last_recurrent_state -> core_out order, co-resident row), the FIRST
  cache-ON vs cache-OFF divergence at the fork step. Reuses the argmax/max_abs
  helpers from scripts/fr13_apc_hit_first_divergence.py when importable.

FAIL-LOUD on: no samples, all-empty, HIT never fired for a cache arm (vacuous),
redirect_used==0 for the refold arm (vacuous refold, §27).

MEMORY: the token reduce is pure-JSON (light). The state reduce streams .pt files
and NEVER holds more than one payload; run it self-demoted under a MemoryMax scope
(--self-demote re-execs under systemd-run when available; the 2026-07-05 OOM lesson).

Read-only. Usage:
    .venv/bin/python seeded2turn_reduce.py --run-root output/fr13_seeded2turn \
        --tokenizer /models/qwen3.6-27b-fp8
    # pass-2 state localization (after windowed capture):
    .venv/bin/python seeded2turn_reduce.py --run-root output/fr13_seeded2turn \
        --state-cache-dir  output/fr13_seeded2turn/cat8_cache/logs/decode_gdn \
        --state-oracle-dir output/fr13_seeded2turn/cat8_nocache/logs/decode_gdn
"""
import argparse, glob, json, os, re, sys
from collections import Counter, OrderedDict

MODEL_EXPECT = "qwen3.6-27b"
REF_ARM = "cat8_nocache"          # A: lossless reference
FLOOR_ARM = "cat8_nocache_b"      # A': floor self-check (2nd nocache boot)
CACHE_ARMS = ["cat8_cache", "cat8_cache_refold"]   # B, C
REFOLD_ARM = "cat8_cache_refold"  # C
ARM_ORDER = [REF_ARM, FLOOR_ARM, "cat8_cache", REFOLD_ARM]

# ----------------------------------------------------------------------------- #
# token-stream extraction
# ----------------------------------------------------------------------------- #
def _load_json(p):
    try:
        return json.load(open(p, errors="ignore"))
    except Exception as e:
        return {"_parse_error": str(e)[:160]}


def choice0(d):
    ch = d.get("choices") or []
    return ch[0] if ch else None


def route_class(ch):
    if not ch:
        return "REQ_ERR"
    m = ch.get("message") or {}
    tcs = m.get("tool_calls") or []
    if not tcs:
        return f"NO_TOOL(finish={ch.get('finish_reason')})"
    name = (tcs[0].get("function") or {}).get("name")
    if name == "agent":
        args = (tcs[0].get("function") or {}).get("arguments") or ""
        return "delegate_agent" + ("[Explore]" if re.search(r"[Ee]xplore", args) else "")
    if name == "read_file":
        return "read_file"
    return f"tool:{name}"


def token_stream(ch, tok=None):
    """Return the EXACT sampled token sequence when logprobs are present (best),
    else a tokenizer re-encoding of reasoning+content (approximate, route_probe
    method). Returns (list_of_token_strings_or_ids, basis_str)."""
    if not ch:
        return [], "empty"
    lp = ch.get("logprobs") or {}
    content_lp = lp.get("content")
    if content_lp:
        toks = [c.get("token") for c in content_lp if c.get("token") is not None]
        if toks:
            return toks, "logprobs"
    m = ch.get("message") or {}
    text = (m.get("reasoning_content") or "") + "\x1e" + (m.get("content") or "")
    # include the tool-call name/args so a pure-route flip still shows a divergence
    for t in (m.get("tool_calls") or []):
        fn = t.get("function") or {}
        text += "\x1e" + str(fn.get("name")) + "\x1e" + str(fn.get("arguments"))[:400]
    if tok is not None:
        try:
            return tok.encode(text, add_special_tokens=False), "retokenize"
        except Exception:
            pass
    return list(text), "chars"


def first_div(a, b):
    """First index at which token lists a,b differ. None if identical up to the
    shorter length AND same length; the shorter length if one is a prefix."""
    if a == b:
        return None
    L = min(len(a), len(b))
    for i in range(L):
        if a[i] != b[i]:
            return i
    return L   # identical prefix, then a length split


# ----------------------------------------------------------------------------- #
# per-arm loading
# ----------------------------------------------------------------------------- #
def load_arm(cdir, tok):
    meta = _load_json(os.path.join(cdir, "arm_meta.json")) if os.path.exists(
        os.path.join(cdir, "arm_meta.json")) else {}
    hits = []
    hp = os.path.join(cdir, "hits.jsonl")
    if os.path.exists(hp):
        for ln in open(hp, errors="ignore"):
            ln = ln.strip()
            if ln:
                try:
                    hits.append(json.loads(ln))
                except Exception:
                    pass
    seeds = {}
    for t in (1, 2):
        for pf in glob.glob(os.path.join(cdir, f"turn{t}_*.json")):
            m = re.search(rf"turn{t}_(\d+)\.json$", pf)
            if not m:
                continue
            k = int(m.group(1))
            d = _load_json(pf)
            ch = choice0(d)
            toks, basis = token_stream(ch, tok)
            seeds.setdefault(k, {})[t] = {
                "route": route_class(ch),
                "toks": toks, "basis": basis,
                "n_tok": len(toks),
                "empty": ch is None or (
                    not ((ch.get("message") or {}).get("tool_calls")
                         or ((ch.get("message") or {}).get("content") or "").strip()
                         or ((ch.get("message") or {}).get("reasoning_content") or "").strip())),
                "finish": (ch or {}).get("finish_reason"),
            }
    return {"meta": meta, "hits": hits, "seeds": seeds}


def dist(vals):
    xs = sorted(v for v in vals if v is not None)
    if not xs:
        return {"n": len(vals), "n_identical": len(vals), "n_diverged": 0}
    return {"n": len(vals), "n_identical": sum(1 for v in vals if v is None),
            "n_diverged": len(xs), "min": xs[0], "median": xs[len(xs)//2], "max": xs[-1]}


# ----------------------------------------------------------------------------- #
# STATE localization (pass-2) — streamed, memory-safe
# ----------------------------------------------------------------------------- #
def state_localize(cache_dir, oracle_dir, argmax_frac_thresh=0.0):
    import torch
    # reuse the proven helpers when the repo module is importable
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                        "..", "..", "..", "..", "..", "..", "scripts"))
        sys.path.insert(0, "/home/mark/shared/lumoFlyWheel/scripts")
        from fr13_apc_hit_first_divergence import _max_abs, _argmax_mismatch  # type: ignore
    except Exception:
        def _max_abs(a, b):
            try:
                a = a.float(); b = b.float()
                if a.shape != b.shape:
                    n = min(a.numel(), b.numel())
                    a = a.reshape(-1)[:n]; b = b.reshape(-1)[:n]
                return float((a - b).abs().max())
            except Exception:
                return None
        def _argmax_mismatch(a, b):
            try:
                x = a.float().reshape(-1, a.shape[-1]); y = b.float().reshape(-1, b.shape[-1])
                n = min(x.shape[0], y.shape[0])
                ax = x[:n].argmax(-1); ay = y[:n].argmax(-1)
                mism = int((ax != ay).sum())
                return {"positions": n, "argmax_mismatch": mism,
                        "frac": (mism / n) if n else 0.0, "max_abs": _max_abs(a, b)}
            except Exception:
                return None

    def index(d):
        # key by (layer_prefix, decode_step) -> path ; stream, never hold two payloads
        out = {}
        for p in sorted(glob.glob(os.path.join(d, "*.pt")) + glob.glob(os.path.join(d, "**/*.pt"), recursive=True)):
            try:
                pay = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            key = (str(pay.get("layer_prefix")), int(pay.get("decode_step", -1)))
            out[key] = p
            del pay
        return out

    on_idx = index(cache_dir)
    or_idx = index(oracle_dir)
    common = sorted(set(on_idx) & set(or_idx), key=lambda k: (k[1], k[0]))  # step, then layer
    if not common:
        return {"error": "no common (layer,step) between cache and oracle state dirs",
                "n_on": len(on_idx), "n_oracle": len(or_idx)}
    findings = []
    for (layer, step) in common:
        try:
            on = torch.load(on_idx[(layer, step)], map_location="cpu", weights_only=False)
            orc = torch.load(or_idx[(layer, step)], map_location="cpu", weights_only=False)
        except Exception as e:
            findings.append({"layer": layer, "step": step, "error": str(e)[:80]})
            continue
        for comp in ("conv_state_rows", "last_recurrent_state", "core_out_spec"):
            a = on.get(comp); b = orc.get(comp)
            if a is None or b is None:
                continue
            ma = _max_abs(a, b)
            am = _argmax_mismatch(a, b) if comp in ("last_recurrent_state", "core_out_spec") else None
            nonzero = (ma is not None and ma > 0.0) or (am and am.get("frac", 0.0) > argmax_frac_thresh)
            if nonzero:
                findings.append({
                    "layer": layer, "step": step, "component": comp,
                    "max_abs": ma, "argmax": am,
                    "coresident_rows": (on.get("coresident_rows").tolist()
                                        if on.get("coresident_rows") is not None else None),
                    # WIRING (layout/row order) vs KERNEL (accumulation) hint:
                    "class_hint": ("WIRING?" if comp == "conv_state_rows" else "KERNEL?"),
                })
        del on, orc
    findings.sort(key=lambda f: (f.get("step", 1 << 30), f.get("layer", "")))
    return {"n_common": len(common), "first_divergences": findings[:40],
            "first": findings[0] if findings else None}


# ----------------------------------------------------------------------------- #
# main
# ----------------------------------------------------------------------------- #
def _load_tokenizer(path):
    if not path:
        return None
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        print(f"[warn] tokenizer '{path}' not loadable ({type(e).__name__}: {str(e)[:80]}); "
              "logprobs-basis streams still work; retokenize fallback -> char", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="output/fr13_seeded2turn")
    ap.add_argument("--tokenizer", default=os.environ.get("SEEDED2TURN_TOKENIZER", ""))
    ap.add_argument("--out", default="")
    ap.add_argument("--state-cache-dir", default="")
    ap.add_argument("--state-oracle-dir", default="")
    ap.add_argument("--argmax-frac-thresh", type=float, default=0.0)
    ap.add_argument("--self-demote", action="store_true",
                    help="re-exec under systemd-run --scope MemoryMax (state reduce OOM guard)")
    ap.add_argument("--memmax", default=os.environ.get("SEEDED2TURN_MEMMAX", "12G"))
    a = ap.parse_args()

    if a.self_demote and not os.environ.get("_S2T_DEMOTED"):
        import shutil, subprocess
        if shutil.which("systemd-run"):
            os.environ["_S2T_DEMOTED"] = "1"
            cmd = ["systemd-run", "--user", "--scope", "-p", f"MemoryMax={a.memmax}",
                   "-p", "MemorySwapMax=0", sys.executable, *sys.argv]
            print(f"[self-demote] re-exec under MemoryMax={a.memmax}", file=sys.stderr)
            sys.exit(subprocess.call(cmd))
        else:
            print("[self-demote] systemd-run not found; continuing UNSCOPED (watch RSS)", file=sys.stderr)

    if not os.path.isdir(a.run_root):
        print(f"FAIL: run-root not found: {a.run_root}", file=sys.stderr); sys.exit(2)
    tok = _load_tokenizer(a.tokenizer)

    arms = [x for x in ARM_ORDER if os.path.isdir(os.path.join(a.run_root, x))]
    arms += [os.path.basename(p) for p in sorted(glob.glob(os.path.join(a.run_root, "*")))
             if os.path.isdir(p) and os.path.basename(p) not in arms
             and os.path.basename(p).startswith("cat8")]
    if not arms:
        print(f"FAIL: no arm dirs under {a.run_root}", file=sys.stderr); sys.exit(2)

    data = {arm: load_arm(os.path.join(a.run_root, arm), tok) for arm in arms}
    report = OrderedDict()
    errors = []

    print("=" * 78)
    print("FR13 2-TURN SEEDED PAIRED-STREAM PROBE — reduce")
    print("=" * 78)

    # ---- per-arm health + route ----
    for arm in arms:
        d = data[arm]; seeds = d["seeds"]; meta = d.get("meta") or {}
        n_seed = len(seeds)
        n_empty = sum(1 for k in seeds for t in (1, 2)
                      if seeds[k].get(t, {}).get("empty", True))
        basis = Counter(seeds[k][t]["basis"] for k in seeds for t in (1, 2) if t in seeds[k])
        r1 = Counter(seeds[k][1]["route"] for k in seeds if 1 in seeds[k])
        r2 = Counter(seeds[k][2]["route"] for k in seeds if 2 in seeds[k])
        obs = (meta.get("obs") or {})
        t2hit = sum(1 for h in d["hits"] if h.get("t2_h", 0) > 0)
        t1hit = sum(1 for h in d["hits"] if h.get("t1_h", 0) > 0)
        print(f"\n### {arm}   (cache_boot={meta.get('cache_boot_log')} mode={meta.get('turn2_mode')} "
              f"seeds={n_seed} empty_turns={n_empty} basis={dict(basis)})")
        print(f"  turn-1 route: {dict(r1)}")
        print(f"  turn-2 route: {dict(r2)}")
        print(f"  hits: turn1_seeds={t1hit} turn2_seeds={t2hit}   obs={obs}")
        # fail-loud health
        if n_seed == 0:
            errors.append(f"{arm}: NO samples")
        if n_seed and n_empty == 2 * n_seed:
            errors.append(f"{arm}: ALL turns empty")
        if arm in CACHE_ARMS and d["hits"] and t2hit == 0:
            errors.append(f"{arm}: turn-2 NEVER hit (t2_h==0 all seeds) => VACUOUS restore/refold probe")
        if arm == REFOLD_ARM and int(obs.get("redirect_used", 0)) == 0:
            errors.append(f"{arm}: redirect_used==0 => REFOLD NEVER EXECUTED; refold A/B is VACUOUS (§27). "
                          "Do NOT read 'refold no help' from this run.")
        report[arm] = {"meta": meta, "route_turn1": dict(r1), "route_turn2": dict(r2),
                       "t1hit_seeds": t1hit, "t2hit_seeds": t2hit, "obs": obs}

    # ---- (2) RESTORE carrier: same-boot turn-1(miss) vs turn-2(hit) ----
    print("\n" + "=" * 78)
    print("(2) RESTORE carrier — SAME-BOOT miss-vs-hit (confound-free; item 1d)")
    print("=" * 78)
    restore = {}
    for arm in [x for x in CACHE_ARMS if x in data]:
        seeds = data[arm]["seeds"]
        idxs = []
        for k in sorted(seeds):
            s = seeds[k]
            if 1 in s and 2 in s and not s[1]["empty"] and not s[2]["empty"]:
                idxs.append(first_div(s[1]["toks"], s[2]["toks"]))
        restore[arm] = dist(idxs)
        print(f"  {arm:<18} turn1-vs-turn2 first-div: {restore[arm]}  "
              f"({restore[arm].get('n_identical',0)}/{restore[arm].get('n',0)} byte-identical)")
    report["restore_carrier"] = restore

    # ---- (1) COLD carrier: cross-boot turn-1 cache-ON vs ref, floor-bracketed ----
    print("\n" + "=" * 78)
    print("(1) COLD carrier — cross-boot TURN-1 cache-ON vs cache-OFF ref (floor = A vs A')")
    print("=" * 78)
    cold = {}
    ref = data.get(REF_ARM, {}).get("seeds", {})
    floor_arm = data.get(FLOOR_ARM, {}).get("seeds", {})
    # floor: A vs A' turn-1
    floor_idx = []
    for k in sorted(set(ref) & set(floor_arm)):
        if 1 in ref[k] and 1 in floor_arm[k] and not ref[k][1]["empty"] and not floor_arm[k][1]["empty"]:
            floor_idx.append(first_div(ref[k][1]["toks"], floor_arm[k][1]["toks"]))
    floor_d = dist(floor_idx)
    floor_min = floor_d.get("min")
    print(f"  FLOOR (A vs A', turn-1): {floor_d}"
          + ("" if floor_arm else "   [!! A' floor arm ABSENT — cross-boot forks are UNATTRIBUTABLE]"))
    if not floor_arm:
        errors.append("FLOOR arm cat8_nocache_b absent: item 1c bracket missing; "
                      "turn-1 token forks cannot be separated from the cross-boot autotune floor.")
    for arm in [x for x in CACHE_ARMS if x in data]:
        seeds = data[arm]["seeds"]
        raw, above = [], 0
        for k in sorted(set(ref) & set(seeds)):
            if 1 in ref[k] and 1 in seeds[k] and not ref[k][1]["empty"] and not seeds[k][1]["empty"]:
                j = first_div(ref[k][1]["toks"], seeds[k][1]["toks"])
                raw.append(j)
                if j is not None and (floor_min is None or j < floor_min):
                    above += 1
        cold[arm] = {"raw": dist(raw), "n_above_floor": above, "floor_min": floor_min}
        print(f"  {arm:<18} turn-1 vs ref: {dist(raw)}   above-floor(earlier than {floor_min}): {above}")
    report["cold_carrier"] = {"floor": floor_d, "arms": cold}

    # ---- (3) REFOLD value ----
    print("\n" + "=" * 78)
    print("(3) REFOLD value — arm C restore-fork vs arm B, + redirect_used liveness")
    print("=" * 78)
    b = restore.get("cat8_cache"); c = restore.get(REFOLD_ARM)
    ru = int((data.get(REFOLD_ARM, {}).get("meta", {}).get("obs") or {}).get("redirect_used", 0))
    verdict = "UNDETERMINED"
    if b and c:
        bmin = b.get("min"); cmin = c.get("min")
        if ru == 0:
            verdict = "VACUOUS (redirect_used==0: refold never executed; C==B by construction)"
        elif c.get("n_identical", 0) == c.get("n", -1) and c.get("n", 0) > 0:
            verdict = "REFOLD MAKES HIT LOSSLESS (arm C turn1==turn2 all seeds)"
        elif bmin is not None and cmin is not None and cmin > bmin:
            verdict = f"REFOLD HELPS (C fork later: {cmin} > {bmin})"
        elif bmin == cmin:
            verdict = "REFOLD INERT (C fork == B fork)"
        else:
            verdict = f"REFOLD WORSE/OTHER (C={cmin} B={bmin})"
    print(f"  arm B restore: {b}")
    print(f"  arm C restore: {c}")
    print(f"  redirect_used (arm C): {ru}")
    print(f"  VERDICT: {verdict}")
    report["refold_verdict"] = {"verdict": verdict, "redirect_used": ru,
                                "B_restore": b, "C_restore": c}

    # ---- STATE localization (pass-2) ----
    if a.state_cache_dir and a.state_oracle_dir:
        print("\n" + "=" * 78)
        print("SECONDARY — per-decode-step STATE localization (pass-2 dumps)")
        print("=" * 78)
        loc = state_localize(a.state_cache_dir, a.state_oracle_dir, a.argmax_frac_thresh)
        report["state_localization"] = loc
        if loc.get("error"):
            print(f"  {loc}")
        else:
            first = loc.get("first")
            print(f"  common (layer,step) pairs: {loc.get('n_common')}")
            if first is None:
                print("  NO state divergence found in the captured window "
                      "(cache-ON == cache-OFF at every captured layer/step/row) — "
                      "either the window missed the fork step D, or the state IS lossless "
                      "and the carrier is downstream (logits/sampler). Widen STEP_LO/HI.")
            else:
                print(f"  FIRST STATE DIVERGENCE: step={first['step']} layer={first['layer']} "
                      f"component={first['component']} class_hint={first['class_hint']}")
                print(f"    max_abs={first.get('max_abs')} argmax={first.get('argmax')}")
                print(f"    coresident_rows={first.get('coresident_rows')}")

    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
        print(f"\n[json] {a.out}")

    if errors:
        print("\n" + "!" * 78)
        print("FAIL — probe validity errors (do NOT interpret results):")
        for e in errors:
            print("  - " + e)
        print("!" * 78)
        sys.exit(1)
    print("\n[ok] probe validity gates passed.")


if __name__ == "__main__":
    main()
