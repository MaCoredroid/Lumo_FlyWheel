#!/usr/bin/env python3
"""FR13 STRUCTURED-OUTPUT CARRIER GATE — dialect-aware generalization of fr13_char8_degeneration_gate.

WHY THIS EXISTS (the false-negative it closes):
  fr13_char8_degeneration_gate.arm_stats() counts turns via `type=='item.completed'` (the CODEX JSONL
  dialect) and char-8 via the vLLM "Unterminated string" 400. The SWE agent is now qwen-code, whose trace
  is a JSON ARRAY ([{type:system},{type:assistant,message:{content:[blocks]}},...,{type:result,num_turns}])
  with ZERO item.completed events -> the char-8 gate (and the replica gate that imports arm_stats) score
  every qwen-code arm as turns=0 / char8=0 => a silent PASS. Meanwhile the real carrier on qwen-code is
  MALFORMED TOOL-CALL MARKUP leaking into a prose text block (verified on cat8 13453: a single text turn
  emits `<tool_search>...</function></tool_call>` unbalanced, tool_calls=0, num_turns=1, empty patch). This
  gate makes the EXPOSURE and the CORRUPTION dialect-aware and unifies char-8-JSON (codex) with
  malformed-XML (qwen-code) into one "did the model emit parseable structured output" rate.

DESIGN (informed by the adversarial verdict on the first spec — FP fixes baked in):
  EXPOSURE (denominator) = agentic turns, dialect-aware:
    codex     : # item.completed events.
    qwen-code : result.num_turns (authoritative; 1 for a turn-1 stall so deaths count) else # assistant
                events carrying a text/tool_use block; min 1 per non-empty trace.
  CORRUPTION (numerator, POOLED emission-integrity defects) = # prose text emissions labelled:
    malformed_markup : tool-call markup (</function>,</tool_call>,<tool_call>,<tool_search>,<parameter=)
                       leaked into a TEXT block instead of being parsed into a tool_use  [THE carrier]
    degenerate_loop  : NON-whitespace char repeated 20+ (FP-FIX: was (.)\\1{20,} which trips 24-space code
                       indent), or table-pipe salad, or doubled-subword / long spaceless run
    off_topic_derail : >20% CJK in a substantial (>20 non-space char) block
  REPORTED SEPARATELY, NOT pooled (per the verdict — each re-imports a different confound):
    silent_drop      : terminal turn with empty text and no tool_use (re-imports the nudge/recovery confound)
    server_parsefail : vLLM "Unterminated string" 400s in docker_full.log (streaming partial-parse noise;
                       verified ANTI-correlated with outcome: native logs MORE than forked -> never pool it)

  TEST: one-sided exact Poisson rate test (conditional-binomial), identical to the char-8 gate, on the
  POOLED committed-corruption count / turns. H0: rate_test <= rate_ref. FAIL(carrier) if p<alpha AND rr>1.

Usage:
  fr13_structured_output_carrier_gate.py <runroot> [--ref <arm|substr>] [--alpha 0.05] [--json]
  fr13_structured_output_carrier_gate.py --ref <arm_dir> --test <arm_dir> [<arm_dir> ...]
  fr13_structured_output_carrier_gate.py --validate <arm_dir>   # per-task label dump (gate self-check)
Read-only, GPU-free. Safe off the DGX.
"""
import argparse, glob, json, os, re, sys
from collections import Counter
from scipy.stats import binomtest

# --- carrier signatures ---
# The reliable tell of leaked tool-call markup in a PROSE text block is an XML CLOSING tag (</function>,
# </tool_call>, </ignore>, </list_directory> ...) or a <parameter ...> tag or an explicit tool open-tag.
# Native prose never contains these (verified POOL=0 across 101 native turns); code snippets use bare `<`
# comparisons ("n < 2") which have no slash, so </\w+> does not FP. Broadened after cat8 13977 miss
# (`<list_directory>...</ignore><parameter name="ignore">` — tool-name tags the vocab list didn't know).
MARKUP = re.compile(r"</[A-Za-z_]\w*>|<\s*parameter\b|<(?:tool_call|tool_response|tool_search|function)\b", re.I)
# infra failures (offload tunnel / socket drops) are NOT a generation defect — exclude from exposure.
INFRA_ERR = re.compile(r"\[API Error|UND_ERR_|terminated \(cause|ECONNRESET|socket hang ?up|Connection reset", re.I)
DEGEN_CHAR = re.compile(r"([^\s])\1{20,}")           # FP-FIX: non-whitespace only (code-indent safe)
DEGEN_TABLE = re.compile(r"(\|[^\n|]*){8,}")          # markdown-table salad
DBL_SUBWORD = re.compile(r"([a-z]{4,})\1")           # understand->underderstand
SPACELESS = re.compile(r"[a-z]{40,}")                # merged words (lowercase run)
CJK = re.compile(r"[　-鿿가-힯぀-ヿ]")
CHAR8_SIG = re.compile(r"Unterminated string starting at")   # vLLM 400 (server-side, reported separate)


def classify_text(txt):
    """Label a single prose text emission. Returns None for empty/whitespace (not a classifiable unit)."""
    if not txt or not txt.strip():
        return None
    t = txt
    if INFRA_ERR.search(t):
        return "infra_error"          # tunnel/socket death — excluded from exposure, not a generation defect
    if MARKUP.search(t):
        return "malformed_markup"
    low = t.lower()
    if DEGEN_CHAR.search(t) or DEGEN_TABLE.search(t) or DBL_SUBWORD.search(low) or SPACELESS.search(low):
        return "degenerate_loop"
    nonspace = [c for c in t if not c.isspace()]
    if len(nonspace) > 20:
        cjk = sum(1 for c in t if CJK.match(c))
        if cjk / len(nonspace) > 0.20:
            return "off_topic_derail"
    return "clean"


def _traces(arm_dir):
    return (glob.glob(os.path.join(arm_dir, "**", "per_task", "*", "agent_trace*.jsonl"), recursive=True)
            + glob.glob(os.path.join(arm_dir, "**", "per_task", "*", "codex_trace*.jsonl"), recursive=True))


def _server_logs(arm_dir):
    seen = dict.fromkeys(glob.glob(os.path.join(arm_dir, "docker_full.log"))
                         + glob.glob(os.path.join(arm_dir, "*docker*full*.log")))
    return list(seen)


def _load(path):
    raw = open(path, errors="ignore").read()
    if raw.lstrip()[:1] == "[":
        try:
            return "qwen", json.loads(raw)
        except Exception:
            return "qwen", []
    objs = []
    for ln in raw.splitlines():
        try:
            objs.append(json.loads(ln))
        except Exception:
            continue
    return "codex", objs


def _trace_labels(dialect, objs):
    """-> (exposure_turns, Counter(labels over text emissions), silent_drop_bool)."""
    labels = Counter()
    texts = []            # (text, has_tool_use_sibling) not needed; keep list of texts in order
    tooluse_turns = 0
    num_turns = None
    if dialect == "qwen":
        for o in objs:
            if not isinstance(o, dict):
                continue
            if o.get("type") == "result":
                num_turns = o.get("num_turns")
            elif o.get("type") == "assistant":
                content = (o.get("message") or {}).get("content")
                blocks = content if isinstance(content, list) else (
                    [{"type": "text", "text": content}] if isinstance(content, str) else [])
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_use":
                        tooluse_turns += 1
                    elif b.get("type") == "text":
                        texts.append(b.get("text") or "")
        # exposure: authoritative num_turns; else observed emissions; min 1 for a non-empty trace.
        obs = tooluse_turns + sum(1 for t in texts if t.strip())
        exposure = num_turns if (isinstance(num_turns, int) and num_turns > 0) else max(obs, 1 if objs else 0)
    else:  # codex JSONL
        for o in objs:
            if isinstance(o, dict) and o.get("type") == "item.completed":
                it = o.get("item") or {}
                if it.get("type") == "agent_message":
                    texts.append(it.get("text") or it.get("message") or "")
                elif it.get("type") in ("function_call", "local_shell_call", "command_execution"):
                    tooluse_turns += 1
        exposure = tooluse_turns + sum(1 for t in texts if t.strip())
    for t in texts:
        lab = classify_text(t)
        if lab:
            labels[lab] += 1
    # infra-death (tunnel/socket): if the trace made no tool call and its only real emission was an infra
    # error, it is NOT a behavioral sample -> exclude from exposure so it can't dilute the corruption rate.
    real_texts = sum(1 for t in texts if t.strip())
    infra_only = (labels.get("infra_error", 0) > 0 and tooluse_turns == 0
                  and real_texts <= labels.get("infra_error", 0))
    if infra_only:
        exposure = 0
    # silent_drop: the LAST non-empty emission is empty/absent AND no tool_use at all (terminal give-up).
    silent_drop = (not infra_only and tooluse_turns == 0 and real_texts == 0 and bool(objs))
    return exposure, labels, silent_drop, int(infra_only)


def arm_stats(arm_dir):
    exposure = 0
    labels = Counter()
    silent_drops = 0
    infra_deaths = 0
    n_traces = 0
    per_task = []
    for f in _traces(arm_dir):
        dialect, objs = _load(f)
        n_traces += 1
        e, lab, sd, infra = _trace_labels(dialect, objs)
        exposure += e
        labels += lab
        silent_drops += int(sd)
        infra_deaths += int(infra)
        task = os.path.basename(os.path.dirname(f))
        per_task.append({"task": task, "dialect": dialect, "turns": e, **dict(lab),
                         "silent_drop": int(sd), "infra_death": int(infra)})
    # server-side parse-fail (reported SEPARATE, never pooled)
    server_pf = 0
    for f in _server_logs(arm_dir):
        try:
            server_pf += sum(1 for ln in open(f, errors="ignore") if CHAR8_SIG.search(ln))
        except Exception:
            pass
    pooled = labels["malformed_markup"] + labels["degenerate_loop"] + labels["off_topic_derail"]
    # resolve
    resolved = graded = 0
    for d in [x for x in glob.glob(os.path.join(arm_dir, "**", "per_task", "*"), recursive=True) if os.path.isdir(x)]:
        for f in glob.glob(os.path.join(d, "**", "eval_report.json"), recursive=True):
            try:
                j = json.load(open(f)); er = j.get("eval_report") or j
                v = er.get("verdict") or ("resolved" if er.get("resolved") else None)
                if v is not None:
                    graded += 1
                    resolved += int(v == "resolved")
            except Exception:
                pass
    return {
        "arm": os.path.basename(arm_dir.rstrip("/")),
        "traces": n_traces, "turns": exposure,
        "malformed_markup": labels["malformed_markup"],
        "degenerate_loop": labels["degenerate_loop"],
        "off_topic_derail": labels["off_topic_derail"],
        "corrupt_pooled": pooled,
        "corrupt_per_1k_turns": round(1000 * pooled / exposure, 2) if exposure else None,
        "silent_drop_SEP": silent_drops,
        "infra_death_SEP": infra_deaths,
        "server_parsefail_SEP": server_pf,
        "resolve": f"{resolved}/{graded}",
        "_per_task": per_task,
    }


def rate_test(c_ref, exp_ref, c_test, exp_test, alpha):
    n = c_ref + c_test
    rr = None
    if exp_ref and exp_test:
        r_ref, r_test = c_ref / exp_ref, c_test / exp_test
        rr = (r_test / r_ref) if r_ref > 0 else (float("inf") if r_test > 0 else 1.0)
    if n == 0 or not exp_ref or not exp_test:
        return 1.0, rr
    p = binomtest(c_test, n, exp_test / (exp_ref + exp_test), alternative="greater").pvalue
    return p, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runroot", nargs="?")
    ap.add_argument("--ref")
    ap.add_argument("--test", nargs="*", default=[])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--validate", help="per-task label dump for one arm dir (gate self-check)")
    a = ap.parse_args()

    if a.validate:
        s = arm_stats(a.validate)
        print(json.dumps({k: v for k, v in s.items()}, indent=2))
        return

    if a.runroot:
        arms = sorted(d for d in glob.glob(os.path.join(a.runroot, "m_*"))
                      if os.path.isdir(d) and not d.endswith(".log"))
        if a.ref and os.path.isdir(a.ref):
            ref_dir = a.ref
        else:
            cand = [d for d in arms if "nat" in os.path.basename(d) or (a.ref and a.ref in d)]
            ref_dir = cand[0] if cand else (arms[0] if arms else None)
        test_dirs = [d for d in arms if d != ref_dir]
    else:
        ref_dir, test_dirs = a.ref, a.test
    if not ref_dir:
        print("no reference arm found", file=sys.stderr); sys.exit(2)

    ref = arm_stats(ref_dir)
    tests = [arm_stats(d) for d in test_dirs]
    rows = [ref] + tests
    results = []
    for t in tests:
        p, rr = rate_test(ref["corrupt_pooled"], ref["turns"], t["corrupt_pooled"], t["turns"], a.alpha)
        verdict = "FAIL(carrier)" if (p < a.alpha and (rr or 0) > 1) else "PASS(behavioral-lossless)"
        results.append({"arm": t["arm"], "rate_ratio_vs_ref": rr, "p_one_sided": p, "verdict": verdict})

    if a.json:
        for r in rows:
            r.pop("_per_task", None)
        print(json.dumps({"ref": ref, "arms": rows, "gate": results}, indent=2)); return
    print(f"==== FR13 STRUCTURED-OUTPUT CARRIER GATE  (ref={ref['arm']}, alpha={a.alpha}) ====")
    hdr = (f"{'arm':<18}{'trc':>4}{'turns':>6}{'mkup':>5}{'degn':>5}{'derl':>5}"
           f"{'POOL':>5}{'/1k':>7}{'|sdrop':>7}{'infra':>6}{'srvPF':>6}{'resolve':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<18}{r['traces']:>4}{r['turns']:>6}{r['malformed_markup']:>5}{r['degenerate_loop']:>5}"
              f"{r['off_topic_derail']:>5}{r['corrupt_pooled']:>5}{str(r['corrupt_per_1k_turns']):>7}"
              f"{r['silent_drop_SEP']:>7}{r['infra_death_SEP']:>6}{r['server_parsefail_SEP']:>6}{r['resolve']:>9}")
    print("  (POOL=malformed_markup+degenerate_loop+off_topic_derail = committed emission defects;")
    print("   sdrop/srvPF reported SEPARATE — NOT pooled: silent_drop=nudge/recovery confound,")
    print("   server_parsefail=streaming partial-parse noise, verified anti-correlated with outcome.)")
    print("\n  GATE vs reference (one-sided exact Poisson rate on POOLED committed-corruption / turn):")
    for g in results:
        rr = g["rate_ratio_vs_ref"]
        rr_s = "inf" if rr == float("inf") else (f"{rr:.2f}x" if rr is not None else "NA")
        print(f"    {g['arm']:<18} rate={rr_s:<7} p={g['p_one_sided']:.4f}  -> {g['verdict']}")


if __name__ == "__main__":
    main()
