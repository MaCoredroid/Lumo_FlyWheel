#!/usr/bin/env python3
"""Build / validate the predeclared q1v3 v3 case set (CPU only).

  python3 cases.py build --trace HARVEST_TRACE.jsonl --requests HARVEST_REQUEST_DIR
        -> writes fixtures/requests/*.json and cases.json (refuses to overwrite)
  python3 cases.py validate -> re-derives every structural invariant of cases.json and its fixture hashes

v3 cases are NATURAL: six fresh prefixes from SWE tasks unseen by v1/v2 and by the replay tuning corpus.
A harvest run of the deployed LumoTree (record-only hooks, deployed sampling) recorded, per decode step,
the 32 tree inputs (root + 31 natural drafts) and the natural TAW outcome (accepted path, emitted tokens).
Cycle k (k < K) of a case = natural step k: its rows, its accepted root-inclusive path and its emitted
pending token. A trailing root-only cycle K consumes the last natural pending token, so that token's
consumption is observed in a state capture (O1_K) and in next-token logits. After cycle K one flush step
consumes z_K (logits only) and emits the terminal token. Every arm is forced through the same stream.
Per prefix three cases share the stream: N (first, cold, common-O0 import), Nb (import, unsalted: primes
the prefix cache and repeats N) and Nr (o0_mode=none: continues from the engine's own prefix-cache
restore -- cache-reuse coverage).
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1v3_common as Q  # noqa: E402

CASES_FILE = os.path.join(HERE, "cases.json")
REQ_DIR = os.path.join(HERE, "fixtures", "requests")
VOCAB = 248320
K_NATURAL = 6            # natural cycles per case (fewer only if a stop token would enter the stream)
K_MIN = 3
NC_PREFIX_COUNT = 2

# Named paths kept for the synthetic (CPU) tests.
P = {"R0": [0], "S1": [0, 1], "S3": [0, 1, 4, 9], "S11": list(Q.SPINE), "B5": [0, 2, 7, 12, 17, 22], "B6": [0, 1, 6],
     "B11": [0, 1, 4, 11], "B13": [0, 3, 8, 13], "B16": [0, 1, 4, 9, 16], "B20": [0, 1, 4, 9, 14, 20], "B21": [0, 1, 4, 9, 14, 21]}
SCHEDULES = {
    "C1-spine-bonus": [("S11", "bonus"), ("R0", "correction"), ("S3", "correction"), ("B21", "correction")],
    "C2-nc-base": [("B6", "correction"), ("S3", "correction"), ("B13", "correction"), ("R0", "correction"), ("S11", "bonus")],
}
NEGATIVE_CONTROLS = [
    ("NC_CONV", "conv history not committed: running-row conv taps restored to their pre-step value", ["conv"]),
    ("NC_GDN", "recurrent replay not committed: running-row SSM state restored to its pre-step value", ["gdn"]),
    ("NC_KV", "attention KV remap off by one row: accepted_paths+1 passed to the KV16 remap only", ["kv"]),
    ("NC_STALE", "stale pending token: next root input overwritten with the previous root (emitted token unchanged)", ["logits"]),
    ("NC_SIB", "sibling-branch commit: accepted_path_rows/last_row = same-length sibling, emitted tokens unchanged", ["kv", "conv", "gdn"]),
]
NC_SIBLING_PATH = [0, 1, 4, 10]
STOP_TOKENS = {Q.TERMINAL_TOKEN, 248044}   # <|im_end|>, <|endoftext|>


def _sha(path):
    return Q.sha256_file(path)


def _rng(*parts):
    return random.Random(int(hashlib.sha256("|".join(["q1v3v3"] + [str(p) for p in parts]).encode()).hexdigest()[:16], 16))


def _draw(rng, pool, n, exclude):
    out, ex = [], set(exclude)
    while len(out) < n:
        t = pool[rng.randrange(len(pool))]
        if t not in ex:
            out.append(t); ex.add(t)
    return out


def _active_children(row):
    return [c for c in Q.children(row) if c not in Q.INACTIVE_ROWS]


def _zkind(nodes, rows, z):
    kids = _active_children(nodes[-1])
    if not kids:
        return "bonus"
    if any(rows[c] == z for c in kids if c not in nodes):
        return "sibling"
    return "correction"


def _assemble(case_id, prefix_key, cycles, flush_rows, extra=None):
    stream, cyc_start = [], []
    for c in cycles:
        cyc_start.append(len(stream))
        stream += [c["row_tokens"][n] for n in c["nodes"]]
    stream.append(cycles[-1]["z"])
    outputs = [cycles[0]["row_tokens"][0]]
    for c in cycles:
        outputs += [c["row_tokens"][n] for n in c["nodes"][1:]] + [c["z"]]
    outputs.append(Q.TERMINAL_TOKEN)
    d = {"case_id": case_id, "prefix": prefix_key, "cycles": cycles, "flush_rows": flush_rows, "stream": stream,
         "cycle_start": cyc_start, "output_tokens": outputs, "max_tokens": len(outputs) + 8, "o0_mode": "import"}
    d.update(extra or {})
    return d


def build_case(case_id, prefix_key, pool, schedule, cycle0_rows=None):
    """Synthetic forced case (CPU tests only)."""
    cycles, root = [], None
    for k, (pname, zkind) in enumerate(schedule):
        nodes = Q.validate_path(P[pname]); rng = _rng(case_id, "cycle", k)
        rows = ([int(t) for t in cycle0_rows] if (k == 0 and cycle0_rows) else (_draw(rng, pool, 32, []) if k == 0 else [root] + _draw(rng, pool, 31, [root])))
        root = rows[0] if k == 0 else root
        kids = Q.children(nodes[-1])
        if zkind == "sibling":
            z = rows[[c for c in kids if c not in nodes][-1]]
        else:
            z = _draw(rng, pool, 1, set(rows))[0]
        cycles.append({"cycle": k, "path_name": pname, "nodes": nodes, "L": len(nodes) - 1, "row_tokens": rows, "z": int(z), "z_kind": zkind,
                       "child_rows_of_last": kids, "source": "synthetic"})
        root = int(z)
    flush = [root] + _draw(_rng(case_id, "flush"), pool, 31, [root])
    return _assemble(case_id, prefix_key, cycles, flush)


def _path_name(nodes):
    for k, v in P.items():
        if v == nodes:
            return k
    return "N" + "-".join(str(n) for n in nodes[1:]) if len(nodes) > 1 else "R0"


def natural_cycles(steps):
    """steps: list of (tree record, taw record) of one request in decode order -> (cycles, flush_rows, notes)."""
    cycles, notes = [], []
    for k, (t, w) in enumerate(steps):
        rows = [int(x) for x in t["ids"]]
        acc = int(w["acc"]); acc_rows = [int(r) for r in w["rows"][:acc]]
        nodes = Q.validate_path([0] + acc_rows)
        toks = [int(x) for x in w["tokens"][: int(w["len"])]]
        if toks[:acc] != [rows[r] for r in acc_rows]:
            notes.append(f"step {k}: TAW tokens != row tokens on the accepted path"); break
        z = toks[acc]
        if k and rows[0] != cycles[-1]["z"]:
            notes.append(f"step {k}: root != previous pending token"); break
        consumed = [rows[n] for n in nodes] + [z]
        if any(x in STOP_TOKENS or not 0 <= x < VOCAB for x in consumed):
            notes.append(f"step {k}: stop/invalid token in the stream; natural cycles end here"); break
        cycles.append({"cycle": k, "path_name": _path_name(nodes), "nodes": nodes, "L": acc, "row_tokens": rows, "z": z,
                       "z_kind": _zkind(nodes, rows, z), "child_rows_of_last": Q.children(nodes[-1]), "source": "natural"})
        if len(cycles) == K_NATURAL:
            break
    if len(cycles) < K_MIN or len(steps) <= len(cycles):
        raise ValueError(f"not enough natural cycles ({len(cycles)}): {notes}")
    # trailing root-only cycle K: consumes the last natural pending token; rows = natural step-K tree
    t, w = steps[len(cycles)]
    rows = [int(x) for x in t["ids"]]
    if rows[0] != cycles[-1]["z"]:
        raise ValueError("trailing cycle root != last pending token")
    z = int(w["tokens"][0])
    if z in STOP_TOKENS:
        z = _draw(_rng("trail", rows[0]), sorted({x for x in rows if x not in STOP_TOKENS and x < Q.SPECIAL_TOKEN_FLOOR}), 1, [])[0]
    cycles.append({"cycle": len(cycles), "path_name": "R0", "nodes": [0], "L": 0, "row_tokens": rows, "z": z,
                   "z_kind": _zkind([0], rows, z), "child_rows_of_last": Q.children(0), "source": "natural-rows/forced-root-only"})
    nxt = steps[len(cycles)][0]["ids"] if len(steps) > len(cycles) else rows
    flush = [z] + [int(x) for x in nxt[1:]]
    return cycles, flush, notes


def sibling_path(nodes):
    if len(nodes) < 3:
        return None
    par, last = nodes[-2], nodes[-1]
    sib = [c for c in _active_children(par) if c != last]
    return (nodes[:-1] + [sib[0]]) if sib else None


def check_case(c):
    cyc = c["cycles"]
    if len(cyc) < 3:
        raise ValueError(f"{c['case_id']}: fewer than 3 cycles")
    s = 0
    for k, x in enumerate(cyc):
        Q.validate_path(x["nodes"])
        if len(x["row_tokens"]) != 32 or any(not 0 <= t < VOCAB for t in x["row_tokens"]):
            raise ValueError(f"{c['case_id']} cycle {k}: bad row tokens")
        if k and x["row_tokens"][0] != cyc[k - 1]["z"]:
            raise ValueError(f"{c['case_id']} cycle {k}: root != previous pending token")
        if c["cycle_start"][k] != s or c["stream"][s:s + x["L"] + 1] != [x["row_tokens"][n] for n in x["nodes"]]:
            raise ValueError(f"{c['case_id']} cycle {k}: stream/cycle_start drift")
        s += x["L"] + 1
    if c["stream"][s:] != [cyc[-1]["z"]] or c["flush_rows"][0] != cyc[-1]["z"] or len(c["flush_rows"]) != 32:
        raise ValueError(f"{c['case_id']}: flush root != final pending token")
    if c["output_tokens"][-1] != Q.TERMINAL_TOKEN or c["output_tokens"][:-1] != c["stream"]:
        raise ValueError(f"{c['case_id']}: emitted tokens != consumed stream + terminal")
    if any(t in STOP_TOKENS for t in c["stream"]):
        raise ValueError(f"{c['case_id']}: stop token inside the stream")
    if c.get("o0_mode", "import") not in ("import", "none"):
        raise ValueError(f"{c['case_id']}: bad o0_mode")
    return True


def _harvest_steps(trace):
    recs = [json.loads(l) for l in open(trace)]
    by_rid, order, cur = {}, [], None
    for r in recs:
        if r["what"] == "tree":
            if r["rid"] not in by_rid:
                by_rid[r["rid"]] = []; order.append(r["rid"])
            cur = [r, None]; by_rid[r["rid"]].append(cur)
        elif r["what"] == "taw" and cur is not None and cur[1] is None:
            cur[1] = r
    return [(rid, [(t, w) for t, w in by_rid[rid] if w is not None]) for rid in order]


def build(trace, req_dir, force=False):
    if os.path.exists(CASES_FILE) and not force:
        raise SystemExit(f"{CASES_FILE} exists (frozen); refuse to overwrite")
    os.makedirs(REQ_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(req_dir) if f.endswith(".json"))
    reqs = _harvest_steps(trace)
    if len(reqs) != len(files):
        raise SystemExit(f"harvest has {len(reqs)} requests, request dir has {len(files)}")
    sources = {"harvest_trace": {"path": trace, "sha256": _sha(trace)}}
    prefixes, cases, nc_candidates = {}, [], []
    for i, (fname, (rid, steps)) in enumerate(zip(files, reqs)):
        key = f"p{i + 1}"
        body = json.load(open(os.path.join(req_dir, fname)))
        req = {"messages": body["messages"], "tools": body.get("tools"), "chat_template_kwargs": body.get("chat_template_kwargs") or {}}
        out = os.path.join(REQ_DIR, f"{key}.json"); json.dump(req, open(out, "w"), sort_keys=True)
        cycles, flush, notes = natural_cycles(steps)
        P0 = int(steps[0][0]["pos0"])
        prefixes[key] = {"source": fname, "block": "evaluation", "held_out": True, "request_file": os.path.relpath(out, HERE),
                         "request_sha256": _sha(out), "reference_prompt_len": P0, "reference_prompt_sha256": "",
                         "harvest_rid": rid, "harvest_steps": len(steps), "notes": notes}
        sources[key] = {"request": {"path": os.path.join(req_dir, fname), "sha256": _sha(os.path.join(req_dir, fname))}}
        for suffix, extra in (("N", {}), ("Nb", {}), ("Nr", {"o0_mode": "none"})):
            cases.append(_assemble(f"{key}__{suffix}", key, [dict(c) for c in cycles], list(flush), extra))
        for k in range(1, len(cycles) - 1):
            sib = sibling_path(cycles[k]["nodes"])
            if sib is not None:
                nc_candidates.append((P0, key, k, sib)); break
    for c in cases:
        check_case(c)
    ncs = []
    for P0, key, k, sib in sorted(nc_candidates)[:NC_PREFIX_COUNT]:
        for name, what, target in NEGATIVE_CONTROLS:
            ncs.append({"nc_id": f"{key}__{name}", "base_case": f"{key}__N", "mutation": name, "cycle": k, "description": what,
                        "must_fail_surfaces_at_cycle": target, "sibling_nodes": sib if name == "NC_SIB" else None})
    if len({n["base_case"] for n in ncs}) < NC_PREFIX_COUNT:
        raise SystemExit("not enough prefixes with a sibling-capable natural cycle for the negative controls")
    doc = {"schema": "q1v3.cases.v3", "sources": sources, "prefixes": prefixes, "cases": cases, "negative_controls": ncs,
           "candidate_repeats": [f"{ncs[0]['base_case']}"], "terminal_token": Q.TERMINAL_TOKEN,
           "topology": {"physical_parent": list(Q.PHYSICAL_PARENT), "inactive_rows": list(Q.INACTIVE_ROWS)}}
    doc["record_sha256"] = Q.record_digest(doc)
    Q.write_json(CASES_FILE, doc)
    print(json.dumps({"cases": len(cases), "negative_controls": len(ncs), "prefixes": len(prefixes), "file": CASES_FILE, "sha256": _sha(CASES_FILE)}))


def load(path=CASES_FILE):
    doc = json.load(open(path))
    if Q.record_digest(doc) != doc["record_sha256"]:
        raise RuntimeError("cases file record digest mismatch")
    return doc


def validate(path=CASES_FILE, check_sources=True):
    doc = load(path)
    for c in doc["cases"]:
        check_case(c)
    for key, p in doc["prefixes"].items():
        if _sha(os.path.join(HERE, p["request_file"])) != p["request_sha256"]:
            raise RuntimeError(f"{key}: request fixture changed")
    if check_sources:
        for key, s in doc["sources"].items():
            for item in ([s] if "path" in s else s.values()):
                if os.path.exists(item["path"]) and _sha(item["path"]) != item["sha256"]:
                    raise RuntimeError(f"{key}: source {item['path']} changed")
    ids = {c["case_id"] for c in doc["cases"]}
    for n in doc["negative_controls"]:
        assert n["base_case"] in ids
    kinds = {}
    for c in doc["cases"]:
        for x in c["cycles"]:
            kinds[x["L"]] = kinds.get(x["L"], 0) + 1
    print(json.dumps({"ok": True, "cases": len(doc["cases"]), "cycles": sum(len(c["cycles"]) for c in doc["cases"]),
                      "negative_controls": len(doc["negative_controls"]), "accepted_length_usage": dict(sorted(kinds.items()))}))
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "validate"]); ap.add_argument("--force", action="store_true")
    ap.add_argument("--trace"); ap.add_argument("--requests")
    a = ap.parse_args()
    build(a.trace, a.requests, a.force) if a.cmd == "build" else validate()
