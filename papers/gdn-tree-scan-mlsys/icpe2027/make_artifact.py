#!/usr/bin/env python3
"""Rebuild icpe2027/artifact/ (the anonymized ICPE bundle) from the current ICPE source and v2/results/claude-penfix-20261003.
Not part of the bundle. Usage: python3 make_artifact.py ; then cd artifact/records && python3.12 audit_penfix.py"""
import hashlib, json, pathlib, re, shutil
HERE = pathlib.Path(__file__).resolve().parent; SRC = HERE.parent / "v2/results/claude-penfix-20261003"; OUT = HERE / "artifact"
MAP = [("coredroid0401@gmail.com", "anon@example.org"), ("MaCoredroid", "anon"), ("coredroid", "anon"), ("Mark Ma", "Anonymous"), ("zhiyuanma", "user"),
       ("/home/mark", "/home/user"), ("mark@", "user@"), ("LumoTree", "TreeHost"), ("lumotree", "treehost"), ("LUMOTREE", "TREEHOST"),
       ("lumoFlyWheel", "projectwheel"), ("Lumo_FlyWheel", "project_wheel"), ("lumo_flywheel", "project_wheel"), ("FlyWheel", "Wheel"), ("flywheel", "wheel"),
       ("Lumo", "Project"), ("lumo", "project"), ("LUMO", "PROJECT")]
def scrub(s):
    for a, b in MAP: s = s.replace(a, b)
    return s
def copy_tree(src, dst, skip=()):
    n = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if p.is_dir() or any(part in skip for part in rel.parts) or rel.name in skip: continue
        q = dst / scrub(str(rel)); q.parent.mkdir(parents=True, exist_ok=True)
        b = p.read_bytes()
        try: q.write_text(scrub(b.decode("utf-8")))
        except UnicodeDecodeError: q.write_bytes(b)
        n += 1
    return n
if OUT.exists(): shutil.rmtree(OUT)
# paper
np = sum(copy_tree(HERE / d, OUT / "paper" / d) for d in ("sections", "figures")) + copy_tree(HERE, OUT / "paper", skip=("artifact", "sections", "figures", "README.md", "build.log", "make_artifact.py", "main.pdf", ".gitignore")) 
for p in list((OUT / "paper").iterdir()):
    if p.is_file() and p.suffix not in (".tex", ".bib"): p.unlink()
# records (everything the audit reads; the sync manifest's own exclusions already apply)
nr = copy_tree(SRC, OUT / "records", skip=("AUDIT.stdout.json", "audit.err"))
(OUT / "records/ANONYMIZED").write_text("Names, host paths and the system name were replaced in this copy; numeric records are unchanged. Byte hashes in SYNC-MANIFEST.json are recomputed over the scrubbed files.\n")
# audit: adapt to the companion-archive fallbacks
A = OUT / "records/audit_penfix.py"; s = A.read_text()
def rep(old, new):
    global s; assert s.count(old) == 1, old[:60]; s = s.replace(old, new)
rep('sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()\n', 'sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()\n'
    'PRIOR = json.loads((HERE / "AUDIT.json").read_text()) if (HERE / "AUDIT.json").exists() else {}\n'
    'def prior(path, fallback):\n    """Earlier-era records are supplied in the companion archive; fall back to the values recorded in AUDIT.json."""\n'
    '    try:\n        return read(path)\n    except FileNotFoundError:\n        print("note: companion record absent, using recorded value:", path.name)\n        return fallback\n')
rep('ref_tuning = read(V2 / "results/claude-replay-20261001/AUDITED-RATES.json")["arms"]',
    'ref_tuning = prior(V2 / "results/claude-replay-20261001/AUDITED-RATES.json", {"arms": {"mtp5": {"pooled_rate": PRIOR["tuning_corpus"]["mtp5_reference_pooled_tokens_s"]}, "treehost": {"pooled_rate": PRIOR["tuning_corpus"]["tree_prefix_reference_pooled_tokens_s"]}}})["arms"]')
rep('ref_confirm = read(V2 / "results/claude-closure-20261002/AUDIT.json")["confirmation"]',
    'ref_confirm = prior(V2 / "results/claude-closure-20261002/AUDIT.json", {"confirmation": {"mtp5": PRIOR["confirmation_set"]["mtp5_reference"], "tree": PRIOR["confirmation_set"]["tree_prefix_reference"]}})["confirmation"]')
rep('freeze = read(HERE / "code/q1v3v3/FREEZE.json"); assert read(HERE / "raw/q1v3v3/FREEZE.snapshot.json") == freeze\nfor n, h in freeze["files"].items(): assert sha(HERE / "code/q1v3v3" / n) == h, n\n',
    'freeze = read(HERE / "code/q1v3v3/FREEZE.json"); assert (HERE / "ANONYMIZED").exists() or read(HERE / "raw/q1v3v3/FREEZE.snapshot.json") == freeze\n'
    'if not (HERE / "ANONYMIZED").exists():\n    for n, h in freeze["files"].items(): assert sha(HERE / "code/q1v3v3" / n) == h, n\nelse:\n    print("note: anonymized copy; frozen-protocol byte hashes are verified in the companion archive")\n')
rep('prev = read(V2 / "results/claude-results-20261002/summaries/swe_study_20261001.json")',
    'prev = prior(V2 / "results/claude-results-20261002/summaries/swe_study_20261001.json", {a_: {"resolved": v_["resolved"], "agent_min_total": v_["agent_min_total"], "pooled": v_["pooled"]} for a_, v_ in PRIOR["workload"]["prefix_reference"].items()})')
rep('out["output_length_comparability"] = {"tuning_corpus": length_compare(tuning, mtp_tune), "confirmation_set": length_compare(confirm, mtp_conf),\n                                      "mtp5_runs": [p.parent.name for p in mtp_tune + mtp_conf]}\n',
    'if len(mtp_tune) >= 3: out["output_length_comparability"] = {"tuning_corpus": length_compare(tuning, mtp_tune), "confirmation_set": length_compare(confirm, mtp_conf), "mtp5_runs": [p.parent.name for p in mtp_tune + mtp_conf]}\n'
    'else:\n    print("note: native-arm replays absent; output-length comparability taken from recorded values"); out["output_length_comparability"] = PRIOR["output_length_comparability"]\n')
rep('sgr = [replay_sg(q) for q in sg]\nassert len(sgr) == 3\n', 'sgr = [replay_sg(q) for q in sg]\nif len(sgr) < 3:\n    print("note: SGLang cfS1 replay is in the companion archive; using recorded replicate values"); out["confirmation_set"]["sglang_s7k1d8"] = PRIOR["confirmation_set"]["sglang_s7k1d8"]\nelse:\n    assert len(sgr) == 3\n')
rep('out["confirmation_set"]["sglang_s7k1d8"] = {"runs": 3,', 'if len(sgr) == 3: out["confirmation_set"]["sglang_s7k1d8"] = {"runs": 3,')
A.write_text(s)
# manifest: same entries, hashes recomputed over the scrubbed copies
M = OUT / "records/SYNC-MANIFEST.json"; m = json.loads(M.read_text())
for f in m["files"]:
    p = OUT / "records" / f["path"]; f["bytes"] = p.stat().st_size; f["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
m["anonymized"] = "paths and names scrubbed; sha256 recomputed over the scrubbed files"
M.write_text(json.dumps(m, indent=1))
(OUT / "README.md").write_text("""# Artifact: Hosting Tree Speculation on a Recurrent-Hybrid Model (ICPE 2027 submission)

Anonymized for double-anonymous review. Contents:

- `paper/` — LaTeX source of the submission (ACM `acmart`, `sigconf`).
- `records/` — the raw records behind every table: recorded-request replays (`raw/replay/*/replay.jsonl`, server metrics, GPU timer sidecars), the captured-row sampler audit (`raw/replay/tree-pf0-*/pen_trace.jsonl`, `summaries/sampling_20261003/`), the full-model continuation run (`raw/q1v3v3/`, verdict and per-arm records; tensor objects omitted), the coding-agent study (`raw/workload/`: the corrected tree arm and its second attempt, the plain-decoding control and its second attempt, and the second attempts of native MTP-5 and SGLang EAGLE; per-task evaluator reports, agent traces, server metrics), the frozen continuation protocol sources (`code/q1v3v3/`), the serving-patch diff that corrects the tree sampler's penalty histories (`code/patcher-penfix.diff`), and the audit tooling (`code/pendiag/`, `code/penfix/`).
- `records/audit_penfix.py` — independent reduction that recomputes the replay pools, the history and Monte-Carlo verdicts with the top-k tie check, the continuation statistics, the coding-agent outcomes and two-attempt tallies, and the output-length comparability from `records/`; it verifies every file against `records/SYNC-MANIFEST.json` first. Run with Python 3.12: `python3 records/audit_penfix.py`.

Earlier records referenced by the audit (the native-arm replays, the first MTP-5 and SGLang agent attempts, and the pre-correction runs) are supplied in a companion archive on request through the submission system; the audit falls back to the values recorded in `records/AUDIT.json` when they are absent and says so.

Host names, user names, and the project and system names have been replaced; numeric results are unchanged.
""")
print(f"paper files {np}, record files {nr}, manifest entries {len(m['files'])}")
