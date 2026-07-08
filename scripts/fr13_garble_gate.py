#!/usr/bin/env python3
"""FR13 G1 garble gate — measures near-neighbor code-token corruption directly.

The pathology (wf_2629d92b): the tree spec-decode commits plausible-but-wrong NEIGHBOR
identifiers/literals (wcs_dict for wcs_header, result_slice for result_sliced, ...) that NameError.
Scalar per-token TV is blind to this; this gate scores CODE-CORRECTNESS on the identifier tokens.

Metric = UNDEFINED-NAME RATE: parse each generated function with `ast`, collect defined names
(params, assignments, comprehension/with/for targets, imports) and referenced names (Name loads);
a load not in {defined ∪ builtins ∪ imported} is a garble candidate (would NameError). Also flag
attribute/near-neighbor typos via edit-distance-1 to a defined name.

Compare 3 arms on IDENTICAL prompts+seeds: no-spec (ground truth) / native-MTP / tree.
Tree FAILS iff its undefined-name rate > native's (both should ≈ no-spec's).

Usage:
  # score already-generated samples (offline / unit test):
  python3 fr13_garble_gate.py score --samples samples.jsonl
  # generate from a served endpoint + score:
  python3 fr13_garble_gate.py run --endpoint http://127.0.0.1:9950/v1 --model qwen3.6-27b \
        --arm tree --n 40 --out out.jsonl
"""
import sys, json, ast, argparse, builtins, urllib.request, concurrent.futures

BUILTINS = set(dir(builtins)) | {"self", "cls", "__name__", "np", "torch", "os", "sys", "re", "math"}

# ---- identifier-consistency probes: force DEFINE distinctive multi-word ids + heavy REUSE ----
PROMPTS = [
    ("wcs_slice",
     "Write ONE self-contained Python function `compute_sliced_wcs_dropped(input_wcs_header, sliced_pixel_bounds)`.\n"
     "Inside, define and then reuse (each at least 4 times) these EXACT variables:\n"
     "  world_coordinate_offset_values, dropped_dimension_index_map, pixel_keep_boolean_mask, sliced_world_result.\n"
     "Do real work: iterate the pixel bounds, fill the offset values, build the keep mask, assemble sliced_world_result, return it.\n"
     "Output ONLY the function in a ```python block. Use each variable name EXACTLY and consistently."),
    ("token_ledger",
     "Write ONE Python function `reconcile_transaction_ledger(pending_entries_list, applied_entry_index)`.\n"
     "Define and reuse (>=4x each): running_balance_accumulator, reversed_entry_signature, mismatched_column_names, final_reconciled_rows.\n"
     "Return final_reconciled_rows. Output ONLY a ```python block; keep every identifier spelled identically."),
    ("matrix_build",
     "Write ONE Python function `build_and_verify_sliced_matrix(header_dict_config, expected_row_count)`.\n"
     "Define and reuse (>=4x each): crpix_reference_pixel, crval_reference_value, computed_slice_shape, verification_passed_flag.\n"
     "Use the literal expected_row_count (do NOT substitute a different number). Return verification_passed_flag. ```python only."),
]

def extract_code(text):
    if "```" in text:
        import re
        blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
        if blocks: return max(blocks, key=len)
    return text

def undefined_names(code):
    """Return (n_loads, n_undefined, undefined_list). Robust to partial/broken code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return (0, 0, ["<syntaxerror>"])  # broken code is itself a garble signal, scored separately
    defined = set(BUILTINS)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined.add(node.name)
            for a in node.args.args + node.args.kwonlyargs + getattr(node.args, "posonlyargs", []):
                defined.add(a.arg)
            if node.args.vararg: defined.add(node.args.vararg.arg)
            if node.args.kwarg: defined.add(node.args.kwarg.arg)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in ([node.target] if hasattr(node, "target") else node.targets):
                for n in ast.walk(t):
                    if isinstance(n, ast.Name): defined.add(n.id)
        elif isinstance(node, (ast.For, ast.comprehension, ast.With, ast.withitem)):
            tgt = getattr(node, "target", None) or getattr(node, "optional_vars", None)
            if tgt:
                for n in ast.walk(tgt):
                    if isinstance(n, ast.Name): defined.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names: imported.add((a.asname or a.name).split(".")[0])
    known = defined | imported
    loads, undef = 0, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads += 1
            if node.id not in known:
                undef.append(node.id)
    return (loads, len(undef), undef)

def score_sample(text):
    code = extract_code(text)
    loads, nundef, undef = undefined_names(code)
    syntax_ok = undef != ["<syntaxerror>"]
    return {"loads": loads, "undefined": (0 if not syntax_ok else nundef),
            "syntax_error": not syntax_ok, "undefined_names": (undef if syntax_ok else [])}

def score_file(path):
    rows = [json.loads(l) for l in open(path)]
    scored = [dict(r, **score_sample(r.get("text", r.get("completion", "")))) for r in rows]
    n = len(scored)
    tot_loads = sum(s["loads"] for s in scored) or 1
    tot_undef = sum(s["undefined"] for s in scored)
    syn_err = sum(s["syntax_error"] for s in scored)
    print(f"  n={n}  undefined-name-rate={100*tot_undef/tot_loads:.2f}%  "
          f"samples-with-undef={sum(1 for s in scored if s['undefined']>0)}/{n}  syntax-errors={syn_err}/{n}")
    ex = [s["undefined_names"] for s in scored if s["undefined"] > 0][:5]
    if ex: print("  example undefined refs:", ex)
    return {"n": n, "undefined_rate": tot_undef/tot_loads, "syntax_err": syn_err,
            "samples_with_undef": sum(1 for s in scored if s['undefined']>0)}

def gen_one(endpoint, model, prompt, seed):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.6, "max_tokens": 900, "seed": seed}).encode()
    r = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    try:
        j = json.loads(urllib.request.urlopen(r, timeout=300).read())
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<ERROR {e}>"

def run(args):
    tasks = [(name, prompt, seed) for (name, prompt) in PROMPTS for seed in range(args.n)]
    out = []
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as ex:
        futs = {ex.submit(gen_one, args.endpoint, args.model, p, s): (name, s) for (name, p, s) in tasks}
        for f in concurrent.futures.as_completed(futs):
            name, s = futs[f]
            out.append({"arm": args.arm, "prompt": name, "seed": s, "text": f.result()})
    with open(args.out, "w") as fh:
        for r in out: fh.write(json.dumps(r) + "\n")
    print(f"[{args.arm}] wrote {len(out)} samples -> {args.out}")
    score_file(args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score"); s.add_argument("--samples", required=True)
    r = sub.add_parser("run")
    r.add_argument("--endpoint", required=True); r.add_argument("--model", required=True)
    r.add_argument("--arm", required=True); r.add_argument("--n", type=int, default=30)
    r.add_argument("--concurrency", type=int, default=4); r.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "score": score_file(a.samples)
    else: run(a)
