#!/usr/bin/env python3
"""FR13 char-8 / degeneration GATE — statistically discriminate char-8 & garble rates between a
REFERENCE arm (native-MTP, the known-clean kernel) and other kernel+cache arms.

This is the BEHAVIORAL complement to the teacher-forced logit gate: char-8 (malformed tool-call JSON)
and degeneration (spaceless/doubled-subword garble) are GENERATION defects that show up in the served
stream, not in fixed-position logits — so we gate them by *rate*, with a denominator (turns) and an
exact statistical test, rather than by eyeballing counts.

Metric (per arm, all tasks pooled):
  turns          = # agentic tool-call turns  (item.completed in codex_trace*.jsonl)  [the EXPOSURE]
  char8          = # vLLM "Unterminated string" 400s in docker_full.log  [SERVER-SIDE, avoids the
                   astropy ecsv.py source-code JSONDecodeError false-positives that live in traces]
  degeneration   = # turns whose agent_message shows the garble signature (doubled subword / long
                   spaceless run)  [best-effort: served tokens for offload runs are remote]
  terminal_char8 = # tasks that FAILED patch_apply/empty AND had char-8 in their turn flow
  resolve        = resolved / graded

Test: two-sample Poisson RATE comparison (exact, conditional-binomial). H0: rate_test <= rate_ref.
  Given c_ref+c_test events, c_test ~ Binomial(c_ref+c_test, turns_test/(turns_ref+turns_test)); the
  one-sided p = P(X >= c_test) is the chance of seeing >= this many char-8s in the test arm if its
  rate were no higher than the reference. FAIL (char-8 carrier) if p < alpha AND rate_ratio > 1.

Usage:
  fr13_char8_degeneration_gate.py <runroot>            # auto-discover arms; ref = the *nativemtp* arm
  fr13_char8_degeneration_gate.py <runroot> --ref m_m_nativemtp5 [--alpha 0.05] [--json]
  fr13_char8_degeneration_gate.py --ref <arm_dir> --test <arm_dir> [<arm_dir> ...]
Read-only. Safe off the DGX.
"""
import argparse, glob, json, os, re, sys
from scipy.stats import binomtest

CHAR8_SIG = re.compile(r"Unterminated string starting at")          # vLLM 400 (server-side)
DBL_SUBWORD = re.compile(r"([a-z]{4,})\1")                          # understand->underderstand
SPACELESS = re.compile(r"[a-z]{35,}")                              # merged words (lowercase run)


def _traces(arm_dir):
    return (glob.glob(os.path.join(arm_dir, "**", "per_task", "*", "agent_trace*.jsonl"), recursive=True)
            + glob.glob(os.path.join(arm_dir, "**", "per_task", "*", "codex_trace*.jsonl"), recursive=True))


def _server_logs(arm_dir):
    # docker_full.log = the vLLM container log (has the char-8 400s). dedup (globs can overlap).
    seen = dict.fromkeys(glob.glob(os.path.join(arm_dir, "docker_full.log"))
                         + glob.glob(os.path.join(arm_dir, "*docker*full*.log")))
    return list(seen)


def _task_dirs(arm_dir):
    return sorted(glob.glob(os.path.join(arm_dir, "**", "per_task", "*"), recursive=True) and
                  [d for d in glob.glob(os.path.join(arm_dir, "**", "per_task", "*"), recursive=True) if os.path.isdir(d)])


def arm_stats(arm_dir):
    turns = 0
    degen_turns = 0
    for f in _traces(arm_dir):
        try:
            for line in open(f, errors="ignore"):
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("type") == "item.completed":
                    turns += 1
                    it = o.get("item") or {}
                    if it.get("type") == "agent_message":
                        txt = (it.get("text") or it.get("message") or "")
                        if DBL_SUBWORD.search(txt.lower()) or SPACELESS.search(txt):
                            degen_turns += 1
        except Exception:
            pass
    # char-8 from the SERVER log (clean); fall back to per-task codex_stderr if no server log
    char8 = 0
    src = "docker_full.log"
    logs = _server_logs(arm_dir)
    if logs:
        for f in logs:
            try:
                char8 += sum(1 for ln in open(f, errors="ignore") if CHAR8_SIG.search(ln))
            except Exception:
                pass
    else:
        src = "codex_stderr (no server log)"
        for f in glob.glob(os.path.join(arm_dir, "**", "agent_stderr*.log"), recursive=True) + glob.glob(os.path.join(arm_dir, "**", "codex_stderr*.log"), recursive=True):
            try:
                char8 += sum(1 for ln in open(f, errors="ignore") if CHAR8_SIG.search(ln))
            except Exception:
                pass
    # resolve + terminal char-8 (task failed patch_apply/empty AND char-8 in its turn flow)
    resolved = graded = terminal_char8 = 0
    for d in [x for x in glob.glob(os.path.join(arm_dir, "**", "per_task", "*"), recursive=True) if os.path.isdir(x)]:
        v = fm = None
        for f in glob.glob(os.path.join(d, "**", "eval_report.json"), recursive=True):
            try:
                j = json.load(open(f)); er = j.get("eval_report") or j
                v = er.get("verdict") or ("resolved" if er.get("resolved") else None)
                fm = er.get("failure_mode")
            except Exception:
                pass
        if v is not None:
            graded += 1
            if v == "resolved":
                resolved += 1
            elif (fm in ("patch_apply_failed", "empty_patch")) or (v in ("patch_apply_failed",)):
                # char-8 present in THIS task's turn flow (stderr, not source)?
                blob = ""
                for sf in glob.glob(os.path.join(d, "agent_stderr*.log")) + glob.glob(os.path.join(d, "codex_stderr*.log")):
                    try:
                        blob += open(sf, errors="ignore").read()
                    except Exception:
                        pass
                if CHAR8_SIG.search(blob):
                    terminal_char8 += 1
    return {
        "arm": os.path.basename(arm_dir.rstrip("/")),
        "turns": turns, "char8": char8, "char8_src": src,
        "char8_per_1k_turns": round(1000 * char8 / turns, 2) if turns else None,
        "degen_turns": degen_turns,
        "degen_per_1k_turns": round(1000 * degen_turns / turns, 2) if turns else None,
        "terminal_char8_fails": terminal_char8,
        "resolve": f"{resolved}/{graded}",
    }


def rate_test(c_ref, exp_ref, c_test, exp_test, alpha):
    """One-sided exact Poisson rate test: is test arm's char-8 rate ELEVATED vs ref? Returns (p, rr)."""
    n = c_ref + c_test
    rr = None
    if exp_ref and exp_test:
        r_ref = c_ref / exp_ref
        r_test = c_test / exp_test
        rr = (r_test / r_ref) if r_ref > 0 else (float("inf") if r_test > 0 else 1.0)
    if n == 0 or not exp_ref or not exp_test:
        return (1.0, rr)
    p_success = exp_test / (exp_ref + exp_test)          # H0: equal rates
    p = binomtest(c_test, n, p_success, alternative="greater").pvalue
    return (p, rr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runroot", nargs="?")
    ap.add_argument("--ref", help="reference arm name (in runroot) or arm dir")
    ap.add_argument("--test", nargs="*", default=[], help="explicit test arm dirs")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    # resolve arm dirs
    if a.runroot:
        arms = sorted(d for d in glob.glob(os.path.join(a.runroot, "m_*"))
                      if os.path.isdir(d) and not d.endswith(".log"))
        if a.ref and os.path.isdir(a.ref):
            ref_dir = a.ref
        else:
            cand = [d for d in arms if "nativemtp" in d or (a.ref and a.ref in d)]
            ref_dir = cand[0] if cand else (arms[0] if arms else None)
        test_dirs = [d for d in arms if d != ref_dir]
    else:
        ref_dir = a.ref
        test_dirs = a.test
    if not ref_dir:
        print("no reference arm found", file=sys.stderr); sys.exit(2)

    ref = arm_stats(ref_dir)
    tests = [arm_stats(d) for d in test_dirs]
    rows = [ref] + tests
    results = []
    for t in tests:
        p, rr = rate_test(ref["char8"], ref["turns"], t["char8"], t["turns"], a.alpha)
        verdict = "FAIL(char8-carrier)" if (p < a.alpha and (rr or 0) > 1) else "PASS(char8-lossless)"
        results.append({"arm": t["arm"], "char8_rate_ratio_vs_ref": rr, "p_one_sided": p, "verdict": verdict})

    if a.json:
        print(json.dumps({"ref": ref, "arms": rows, "gate": results}, indent=2))
        return
    print(f"==== FR13 char-8 / degeneration GATE   (ref = {ref['arm']}, alpha={a.alpha}) ====")
    hdr = f"{'arm':<20}{'turns':>7}{'char8':>7}{'/1k-turn':>9}{'degen/1k':>9}{'termC8':>7}{'resolve':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<20}{r['turns']:>7}{r['char8']:>7}{str(r['char8_per_1k_turns']):>9}"
              f"{str(r['degen_per_1k_turns']):>9}{r['terminal_char8_fails']:>7}{r['resolve']:>9}")
    print(f"  (char8 source: {ref['char8_src']})")
    print("\n  GATE vs reference (one-sided exact Poisson rate test on char-8/turn):")
    for g in results:
        rr = g["char8_rate_ratio_vs_ref"]
        rr_s = "inf" if rr == float("inf") else (f"{rr:.2f}x" if rr is not None else "NA")
        print(f"    {g['arm']:<20} rate={rr_s:<7} p={g['p_one_sided']:.4f}  -> {g['verdict']}")


if __name__ == "__main__":
    main()
