#!/usr/bin/env python3
"""Build / validate the predeclared q1v3 case set (CPU only).

  python3 cases.py build    -> writes fixtures/requests/*.json and cases.v1.json (refuses to overwrite)
  python3 cases.py validate -> re-derives every structural invariant of cases.v1.json and its fixture hashes

Prefixes: the six Codex Q1 prefixes (3 calibration, 3 held-out evaluation; chat bodies from
prefix-plan/selected/*/replay-input.json) plus one block-boundary prefix from the v2exp corpus whose
rendered length is 65 tokens below a 1024 multiple (25535), so multi-cycle commits cross a
KV-page / mamba-block boundary.

Each case is a multi-cycle forced stream. Cycle k: root r_k (= z_{k-1}), 31 forced draft rows, a forced
accepted root-inclusive path, and a forced pending token z_k. After the last cycle one flush step
consumes z_{K-1} (observing its next-token logits) and is forced to emit the terminal token.
Cycle-0 tree rows of Codex prefixes reuse Codex's token-fixtures.v1 assignment; every other forced
token is drawn deterministically (sha256-seeded) from the prefix's distinct non-special tokens.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, struct, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q1v3_common as Q  # noqa: E402

CODEX = "/home/mark/lumotree-review-20260927/papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927"
CODEX_FIXTURES = f"{CODEX}/fullmodel/fixtures/token-fixtures.v1.json"
CODEX_SELECTED = f"{CODEX}/prefix-plan/selected"
CORPUS = "/home/mark/shared/lumotree-v2exp-runs/corpus"
BOUNDARY_REQUEST = "chatreq_1787598990204.json"      # rendered 25535 tokens; 25600 = 25 * 1024
CASES_FILE = os.path.join(HERE, "cases.v1.json")
REQ_DIR = os.path.join(HERE, "fixtures", "requests")

PREFIXES = [
    ("cal-short", "calibration-short_available", "calibration", False),
    ("cal-medium", "calibration-medium_available", "calibration", False),
    ("cal-long", "calibration-long_available", "calibration", False),
    ("eval-short", "evaluation-short_available", "evaluation", True),
    ("eval-medium", "evaluation-medium_available", "evaluation", True),
    ("eval-long", "evaluation-long_available", "evaluation", True),
]

# Named root-inclusive paths (physical rows).  S = spine, B = off-spine branch.
P = {
    "R0": [0],
    "S1": [0, 1],
    "S3": [0, 1, 4, 9],
    "S11": list(Q.SPINE),                       # max depth 11, leaf -> bonus
    "B5": [0, 2, 7, 12, 17, 22],                # off-spine chain ending at a leaf
    "B6": [0, 1, 6],
    "B11": [0, 1, 4, 11],
    "B13": [0, 3, 8, 13],                       # node 13's only child (row 18) is an inactive padding row
    "B16": [0, 1, 4, 9, 16],
    "B20": [0, 1, 4, 9, 14, 20],
    "B21": [0, 1, 4, 9, 14, 21],
    "S5": [0, 1, 4, 9, 14, 19],
    "B7L4": [0, 1, 4, 9, 15],
}
# (path name, pending-token kind).  sibling = z equals the token of an UNACCEPTED child of the last node.
SCHEDULES = {
    "C1-spine-bonus": [("S11", "bonus"), ("R0", "correction"), ("S3", "correction"), ("B21", "correction")],
    "C2-nc-base": [("B6", "correction"), ("S3", "correction"), ("B13", "correction"), ("R0", "correction"), ("S11", "bonus")],
    "C3-branch-leaf": [("B5", "bonus"), ("S3", "sibling"), ("B11", "correction"), ("S1", "correction")],
    "C4-root-only": [("R0", "correction"), ("R0", "correction"), ("R0", "sibling"), ("R0", "correction")],
    "C5-deep-branch": [("B20", "correction"), ("S11", "bonus"), ("S1", "sibling"), ("B13", "correction")],
}
# Boundary prefix (P = 25535, boundary 25600 = P + 65).
BOUNDARY_SCHEDULES = {
    # cycle 5 commits P+60..P+71: the boundary falls inside an accepted path
    "K1-cross-inside": [("S11", "bonus")] * 5 + [("S11", "bonus"), ("R0", "correction")],
    # cycle 5 commits P+60..P+65: the last accepted draft is the first token of the new block
    "K2-cross-last-draft": [("S11", "bonus")] * 5 + [("B5", "bonus"), ("S3", "correction")],
    # cycle 5 commits P+60..P+64: the pending token z_5 sits at the block start (next root at the boundary)
    "K3-pending-at-boundary": [("S11", "bonus")] * 5 + [("B16", "correction"), ("B6", "correction")],
}
NEGATIVE_CONTROLS = [  # applied to the C2-nc-base stream at cycle 1 (path S3, sibling S3' = [0,1,4,10])
    ("NC_CONV", "conv history not committed: running-row conv taps restored to their pre-step value", ["conv"]),
    ("NC_GDN", "recurrent replay not committed: running-row SSM state restored to its pre-step value", ["gdn"]),
    ("NC_KV", "attention KV remap off by one row: accepted_paths+1 passed to the KV16 remap only", ["kv"]),
    ("NC_STALE", "stale pending token: next root input overwritten with the previous root (emitted token unchanged)", ["logits"]),
    ("NC_SIB", "sibling-branch commit: accepted_path_rows/last_row = same-length sibling, emitted tokens unchanged", ["kv", "conv", "gdn"]),
]
NC_PREFIXES = ["cal-short", "cal-medium"]
NC_CYCLE = 1
NC_SIBLING_PATH = [0, 1, 4, 10]
CANDIDATE_REPEATS = ["cal-short__C2-nc-base"]   # clean repeat in the candidate process (determinism diagnostic)


def _sha(path):
    return Q.sha256_file(path)


def _pool(ids):
    return sorted({int(t) for t in ids if 0 <= int(t) < Q.SPECIAL_TOKEN_FLOOR})


def _rng(*parts):
    return random.Random(int(hashlib.sha256("|".join(["q1v3"] + [str(p) for p in parts]).encode()).hexdigest()[:16], 16))


def _draw(rng, pool, n, exclude):
    out = []
    ex = set(exclude)
    while len(out) < n:
        t = pool[rng.randrange(len(pool))]
        if t not in ex:
            out.append(t); ex.add(t)
    return out


def build_case(case_id, prefix_key, pool, schedule, cycle0_rows=None):
    cycles = []
    root = None
    for k, (pname, zkind) in enumerate(schedule):
        nodes = Q.validate_path(P[pname])
        rng = _rng(case_id, "cycle", k)
        if k == 0:
            if cycle0_rows is not None:
                rows = [int(t) for t in cycle0_rows]
            else:
                rows = _draw(rng, pool, 32, [])
            root = rows[0]
        else:
            rows = [root] + _draw(rng, pool, 31, [root])
        last = nodes[-1]
        kids = Q.children(last)
        if zkind == "bonus" and kids:
            raise ValueError(f"{case_id} cycle {k}: bonus pending requires a leaf, node {last} has children {kids}")
        if zkind == "sibling":
            unacc = [c for c in kids if c not in nodes]
            if not unacc:
                raise ValueError(f"{case_id} cycle {k}: no unaccepted child for a sibling-token pending")
            z = rows[unacc[-1]]
        else:
            z = _draw(rng, pool, 1, set(rows))[0]
        cycles.append({"cycle": k, "path_name": pname, "nodes": nodes, "L": len(nodes) - 1, "row_tokens": rows,
                       "z": int(z), "z_kind": zkind, "child_rows_of_last": kids})
        root = int(z)
    flush_rows = [root] + _draw(_rng(case_id, "flush"), pool, 31, [root])
    stream, cyc_start = [], []
    for c in cycles:
        cyc_start.append(len(stream))
        stream += [c["row_tokens"][n] for n in c["nodes"]]          # root + accepted drafts (consumed this cycle)
    stream.append(cycles[-1]["z"])                                    # final pending token, consumed by the flush step
    outputs = [cycles[0]["row_tokens"][0]]
    for c in cycles:
        outputs += [c["row_tokens"][n] for n in c["nodes"][1:]] + [c["z"]]
    outputs.append(Q.TERMINAL_TOKEN)
    return {"case_id": case_id, "prefix": prefix_key, "cycles": cycles, "flush_rows": flush_rows,
            "stream": stream, "cycle_start": cyc_start, "output_tokens": outputs,
            "max_tokens": len(outputs) + 8}


def check_case(c):
    """Structural invariants (also run by the hooks at boot and by the tests)."""
    cyc = c["cycles"]
    if len(cyc) < 3:
        raise ValueError(f"{c['case_id']}: fewer than 3 cycles")
    s = 0
    for k, x in enumerate(cyc):
        Q.validate_path(x["nodes"])
        if len(x["row_tokens"]) != 32 or any(not 0 <= t < Q.SPECIAL_TOKEN_FLOOR for t in x["row_tokens"]):
            raise ValueError(f"{c['case_id']} cycle {k}: bad row tokens")
        if k and x["row_tokens"][0] != cyc[k - 1]["z"]:
            raise ValueError(f"{c['case_id']} cycle {k}: root != previous pending token")
        if c["cycle_start"][k] != s:
            raise ValueError(f"{c['case_id']} cycle {k}: cycle_start drift")
        if c["stream"][s:s + x["L"] + 1] != [x["row_tokens"][n] for n in x["nodes"]]:
            raise ValueError(f"{c['case_id']} cycle {k}: stream != root+accepted drafts")
        s += x["L"] + 1
    if c["stream"][s:] != [cyc[-1]["z"]] or c["flush_rows"][0] != cyc[-1]["z"]:
        raise ValueError(f"{c['case_id']}: flush root != final pending token")
    if c["output_tokens"][-1] != Q.TERMINAL_TOKEN or c["output_tokens"][:-1] != c["stream"]:
        raise ValueError(f"{c['case_id']}: emitted tokens != consumed stream + terminal")
    if Q.TERMINAL_TOKEN in c["stream"]:
        raise ValueError(f"{c['case_id']}: terminal token inside the stream")
    return True


def build(force=False):
    if os.path.exists(CASES_FILE) and not force:
        raise SystemExit(f"{CASES_FILE} exists (frozen); refuse to overwrite")
    os.makedirs(REQ_DIR, exist_ok=True)
    fx = json.load(open(CODEX_FIXTURES)); fxp = {p["prefix_id"]: p for p in fx["prefixes"]}
    sources = {"codex_token_fixtures": {"path": CODEX_FIXTURES, "sha256": _sha(CODEX_FIXTURES)}}
    prefixes, cases = {}, []
    for key, pid, block, held in PREFIXES:
        rin = os.path.join(CODEX_SELECTED, pid, "replay-input.json"); tok = os.path.join(CODEX_SELECTED, pid, "token-ids.u32le")
        body = json.load(open(rin))
        req = {"messages": body["messages"], "tools": body.get("tools"), "chat_template_kwargs": body.get("chat_template_kwargs") or {}}
        out = os.path.join(REQ_DIR, f"{key}.json"); json.dump(req, open(out, "w"), sort_keys=True)
        raw = open(tok, "rb").read(); ids = list(struct.unpack(f"<{len(raw)//4}I", raw))
        pool = _pool(ids)
        prefixes[key] = {"source": pid, "block": block, "held_out": held, "request_file": os.path.relpath(out, HERE), "request_sha256": _sha(out),
                         "reference_prompt_len": len(ids), "reference_prompt_sha256": hashlib.sha256(raw).hexdigest(), "pool_size": len(pool)}
        sources[key] = {"replay_input": {"path": rin, "sha256": _sha(rin)}, "token_ids": {"path": tok, "sha256": _sha(tok)}}
        f0 = fxp[pid]
        if hashlib.sha256(raw).hexdigest() != f0["token_ids_sha256"]:
            raise SystemExit(f"{pid}: token ids sha != Codex fixture")
        for sname, sched in SCHEDULES.items():
            cases.append(build_case(f"{key}__{sname}", key, pool, sched, cycle0_rows=f0["physical_row_tokens"]))
    # block-boundary prefix from the v2exp corpus
    breq = os.path.join(CORPUS, "requests", BOUNDARY_REQUEST); bids_file = os.path.join(CORPUS, "vllm_prompt_ids.json")
    body = json.load(open(breq)); ids = json.load(open(bids_file))[BOUNDARY_REQUEST]["ids"]
    req = {"messages": body["messages"], "tools": body.get("tools"), "chat_template_kwargs": body.get("chat_template_kwargs") or {}}
    out = os.path.join(REQ_DIR, "bnd-25535.json"); json.dump(req, open(out, "w"), sort_keys=True)
    raw = struct.pack(f"<{len(ids)}I", *ids)
    prefixes["bnd-25535"] = {"source": BOUNDARY_REQUEST, "block": "boundary", "held_out": True, "request_file": os.path.relpath(out, HERE),
                             "request_sha256": _sha(out), "reference_prompt_len": len(ids), "reference_prompt_sha256": hashlib.sha256(raw).hexdigest(),
                             "pool_size": len(_pool(ids)), "block_boundary": (len(ids) // 1024 + 1) * 1024}
    sources["bnd-25535"] = {"request": {"path": breq, "sha256": _sha(breq)}, "vllm_prompt_ids": {"path": bids_file, "sha256": _sha(bids_file)}}
    for sname, sched in BOUNDARY_SCHEDULES.items():
        cases.append(build_case(f"bnd-25535__{sname}", "bnd-25535", _pool(ids), sched))
    for c in cases:
        check_case(c)
    by_id = {c["case_id"]: c for c in cases}
    ncs = []
    for key in NC_PREFIXES:
        base = by_id[f"{key}__C2-nc-base"]
        assert base["cycles"][NC_CYCLE]["nodes"] == P["S3"]
        for name, what, target in NEGATIVE_CONTROLS:
            ncs.append({"nc_id": f"{key}__{name}", "base_case": base["case_id"], "mutation": name, "cycle": NC_CYCLE, "description": what,
                        "must_fail_surfaces_at_cycle": target, "sibling_nodes": NC_SIBLING_PATH if name == "NC_SIB" else None})
    doc = {"schema": "q1v3.cases.v1", "sources": sources, "prefixes": prefixes, "paths": P, "schedules": SCHEDULES,
           "boundary_schedules": BOUNDARY_SCHEDULES, "cases": cases, "negative_controls": ncs, "candidate_repeats": CANDIDATE_REPEATS,
           "terminal_token": Q.TERMINAL_TOKEN, "topology": {"physical_parent": list(Q.PHYSICAL_PARENT), "inactive_rows": list(Q.INACTIVE_ROWS)}}
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
            kinds.setdefault(x["path_name"], 0); kinds[x["path_name"]] += 1
    print(json.dumps({"ok": True, "cases": len(doc["cases"]), "cycles": sum(len(c["cycles"]) for c in doc["cases"]),
                      "negative_controls": len(doc["negative_controls"]), "path_usage": kinds}))
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "validate"]); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build(a.force) if a.cmd == "build" else validate()
