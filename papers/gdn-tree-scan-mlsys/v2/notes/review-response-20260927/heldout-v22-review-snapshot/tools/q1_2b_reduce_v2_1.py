#!/usr/bin/env python3
"""Q1.2b reducer **v2.1** (= v2 with the runner helper identity renamed to q1_component_runner_v2_1.py; no other change): source-bound, both-process, raw-witness reduction (reducer review R1-R4 + parent runner review).

Inputs (all inside the run directory, written by the launcher/runner): LAUNCH-BINDING.json (hash-bound expected
identities/scope/run id copied from the approved gate at launch), snapshots of the fixture manifest / expected
observations / policy v2.1 / gate, procA/ and procB/ (result.json + raw_inventory.jsonl + ref/ eligibility/ raw/),
and the shared content-addressed tensor store.

Order (each stage fails closed):
  R1  binding: LAUNCH-BINDING present and hashes recomputed from the snapshot bytes; expected block/repeats/process set;
      canonical payloads of manifest + observations recomputed; typed unique fixture/case keys; exact full
      selected-block product; every mandatory attestation identity present and EQUAL TO THE BOUND VALUE (not merely
      equal across processes); process results eligible/complete/integrity/block/repeats/counters consistent.
  R2  raw inventories: per process read and hashed; every expected case has exactly one `ref`, one `eligibility` and one
      `metrics` record per process; record file sha + record_sha256 recomputed; identity fields present; candidate /
      native / C2 hashes bound to the expected reference sha and to existing tensor files (byte length + sha verified);
      B's references reconciled with A's (native hashes equal) -- never inferred from candidate identity.
  R3  repeats: exact repeat index set {0..R-1}; nonempty scan/publish maps with exactly the expected paths; within-
      and cross-process candidate AND native determinism derived from raw hashes; per-publication flags reconciled.
  R4  negatives (calibration, BOTH processes, EVERY fixture): exact tag set, exact record product (N1 L, N3 2, N4 L,
      N5 L), integer counters, boolean flags, N5 = poisoned C0 == clean C0 bitwise (hash equality recomputed from
      records), N7 rejects; then numerical power N1/N3/N4 FAIL under the paired rules (N5 paired result reported only).
  N   numerical: candidate-blind eligibility recomputed from the sealed ref records (both processes), evaluation with
      evaluator v2.1 over the metrics records; aggregate over the FIXED expected case set; optional recomputation of all
      metric arrays from the retained tensors + CPU-regenerated C2 (--recompute all).
Exit codes: 2 malformed/ineligible/unbound; 5 structural/determinism/negative failure; 6 numerical FAIL; 4 UNCOVERED; 0 PASS.
"""
import argparse, hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import q1_policy_evaluator_v2_1 as PE21  # noqa: E402

STRUCT_KEYS = ("c2_hash_matches_fixture", "c1_within_process_bitwise", "within_process_bitwise", "ring_bytes_exact_all", "replays_one_per_publish", "pointer_identity_unchanged",
               "control_rows_intact", "staging_neutral_tail", "no_nonfinite_candidate", "reference_phase_before_candidate", "expected_case_product_sealed",
               "n5_poisoned_equals_clean_candidate", "negative_product_exact")
NEG_TAGS = ("N1_sibling_substitution", "N3_ring_swap", "N4_stale_metadata", "N5_offpath_sentinels", "N7_incumbent_twice")
MANDATORY_ATT = ("kernel_module_sha256", "topology_sha256", "native_module_sha256", "native_function_source_sha256", "runner_sha256", "oracle_sha256", "fixtures_tool_sha256",
                 "image_id_arg", "torch", "triton", "gpu_name", "policy_sha256", "expected_observations_sha256", "fixture_manifest_sha256", "block", "repeats", "helper_hashes",
                 "hostname_in_container", "container_name_env", "utc", "pid", "vllm")
EXPECT_KEYS = ("image_id", "fixture_manifest", "expected_observations", "policy", "policy_base_v2", "contract_v2", "runner", "reducer", "launcher", "fixtures_tool", "oracle",
               "evaluator_v1", "evaluator_v2", "evaluator_v2_1", "bf16_ulp", "topology", "kernel", "native_module", "verification_V2", "tree_conv_fused", "gate")
HELPER_KEYS = ("q1_component_runner_v2_1.py", "q1_oracle.py", "q1_2b_fixtures.py", "q1_policy_evaluator.py", "q1_policy_evaluator_v2.py", "q1_policy_evaluator_v2_1.py", "q1_bf16_ulp.py",
               "scripts/fr13_fixed32_topology.py", "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py", "src/lumo_flywheel_serving/fr13_tree_conv_fused.py")
HELPER_TO_EXPECT = {"q1_component_runner_v2_1.py": "runner", "q1_oracle.py": "oracle", "q1_2b_fixtures.py": "fixtures_tool", "q1_policy_evaluator.py": "evaluator_v1",
                    "q1_policy_evaluator_v2.py": "evaluator_v2", "q1_policy_evaluator_v2_1.py": "evaluator_v2_1", "q1_bf16_ulp.py": "bf16_ulp",
                    "scripts/fr13_fixed32_topology.py": "topology", "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": "kernel", "src/lumo_flywheel_serving/fr13_tree_conv_fused.py": "tree_conv_fused"}
AB_RUNTIME_KEYS = ("torch", "triton", "gpu_name", "vllm", "image_id_arg", "torch_cuda")
TENSOR_SHAPES = {"state": [48, 128, 128], "output": [48, 128]}
DTYPE_BYTES = {"torch.bfloat16": 2, "torch.float32": 4}


def reducer_runtime_identity():
    """The environment this reduction actually runs in (must be the pinned image, CPU-only, single-threaded)."""
    import platform, socket
    rt = {"python": platform.python_version(), "hostname": socket.gethostname(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
          "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "container_name_env": os.environ.get("Q12B_CONTAINER_NAME"), "pid": os.getpid()}
    try:
        import torch
        rt.update({"torch": torch.__version__, "torch_threads": torch.get_num_threads(), "cuda_available": bool(torch.cuda.is_available())})
    except Exception as e:  # noqa: BLE001
        rt.update({"torch": None, "torch_error": type(e).__name__})
    return rt


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def _load_json(p, F, label):
    try:
        return json.load(open(p))
    except Exception as e:  # noqa: BLE001
        F["malformed"].append(f"{label}: unreadable/invalid JSON ({type(e).__name__})"); return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _case_rel(case_id):
    fid, rest = case_id.split("|", 1)
    return fid.replace("/", "__"), rest.replace("|", "__") + ".json"


def process_independence(run_dir, att_a, att_b):
    """Two fresh containers both report os.getpid()==1 (separate PID namespaces); independence is bound to distinct container
    identity (in-container hostname == docker container id prefix, launcher-bound container name), distinct process tags and
    distinct start clocks. PID is recorded as evidence only and may legitimately be equal."""
    pr = []
    for k in ("hostname_in_container", "container_name_env", "process_tag", "utc"):
        va, vb = att_a.get(k), att_b.get(k)
        if not isinstance(va, str) or not va or not isinstance(vb, str) or not vb:
            pr.append(f"{k} missing in an attestation")
        elif va == vb:
            pr.append(f"{k} identical across processes ({va!r})")
    if att_a.get("process_tag") != "A" or att_b.get("process_tag") != "B":
        pr.append("process tags are not A/B")
    for tag, att in (("A", att_a), ("B", att_b)):
        if att.get("container_name_env") not in (None, f"lumotree-review-q12b-{tag}"):
            pr.append(f"proc{tag}: container name env {att.get('container_name_env')!r} != launcher-bound name")
        cid_p = os.path.join(run_dir, f"proc{tag}.cid")
        if os.path.exists(cid_p):
            cid = open(cid_p).read().strip()
            if not cid or not str(att.get("hostname_in_container", "")).startswith(cid[:12]):
                pr.append(f"proc{tag}: in-container hostname does not match the launcher-bound container id")
    return {"independent": not pr, "problems": pr, "pid_A": att_a.get("pid"), "pid_B": att_b.get("pid"), "pid_equal_allowed": True}


def load_inventory(run_dir, tag, F):
    p = os.path.join(run_dir, f"proc{tag}", "raw_inventory.jsonl")
    if not os.path.exists(p):
        F["malformed"].append(f"proc{tag}: raw_inventory.jsonl missing"); return None, None
    entries = []
    with open(p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:  # noqa: BLE001
                F["malformed"].append(f"proc{tag}: inventory line {i} invalid"); return None, None
    return entries, sha256_file(p)


def verify_record(run_dir, tag, entry, F, tensors_root, seen_tensor, require_tensors=False, surface=None):
    """Recompute file sha + record sha, load the record, verify its tensor refs exist with the right bytes/sha.
    For metrics/negative records both tensor refs are mandatory and must carry a declared dtype, the canonical shape and byte length."""
    p = os.path.join(run_dir, f"proc{tag}", entry["path"])
    if not os.path.exists(p):
        F["malformed"].append(f"proc{tag}: record file missing {entry['path']}"); return None
    if sha256_file(p) != entry.get("file_sha256") or os.path.getsize(p) != entry.get("bytes"):
        F["malformed"].append(f"proc{tag}: record file sha/bytes differ from inventory {entry['path']}"); return None
    rec = _load_json(p, F, f"proc{tag}:{entry['path']}")
    if rec is None:
        return None
    body = dict(rec); rsha = body.pop("record_sha256", None)
    if rsha != canonical_sha(body) or rsha != entry.get("record_sha256"):
        F["malformed"].append(f"proc{tag}: record_sha256 mismatch {entry['path']}"); return None
    if entry.get("case_id") != rec.get("case_id"):
        F["malformed"].append(f"proc{tag}: {entry['path']} record case_id != inventory case_id"); return None
    for key in ("candidate_tensor", "native_tensor"):
        t = rec.get(key)
        if t is None:
            if require_tensors:
                F["malformed"].append(f"proc{tag}: {entry['path']} {key} missing (mandatory)"); return None
            continue
        if not isinstance(t, dict) or not isinstance(t.get("sha256"), str) or not _is_int(t.get("bytes")) or t.get("dtype") not in DTYPE_BYTES or not isinstance(t.get("shape"), list):
            F["malformed"].append(f"proc{tag}: {entry['path']} {key} malformed"); return None
        surf = surface or rec.get("surface")
        want_shape = TENSOR_SHAPES.get(surf)
        n = 1
        for d in t["shape"]:
            n *= int(d)
        if want_shape is not None and (list(t["shape"]) != want_shape or t["bytes"] != n * DTYPE_BYTES[t["dtype"]]):
            F["malformed"].append(f"proc{tag}: {entry['path']} {key} shape/bytes {t['shape']}/{t['bytes']} not canonical for {surf}"); return None
        tp = os.path.join(tensors_root, t["sha256"] + ".bin")
        if t["sha256"] not in seen_tensor:
            if not os.path.exists(tp) or os.path.getsize(tp) != t["bytes"] or sha256_file(tp) != t["sha256"]:
                F["malformed"].append(f"proc{tag}: tensor {t['sha256'][:12]} missing/corrupt for {entry['path']}"); return None
            seen_tensor.add(t["sha256"])
        if rec.get(key.replace("_tensor", "_sha256")) != t["sha256"]:
            F["malformed"].append(f"proc{tag}: {entry['path']} {key} sha field inconsistent"); return None
    return rec


def reduce_run(run_dir, block="calibration", recompute="none", tensors_root=None):
    F = {"malformed": [], "structural": [], "numerical": [], "uncovered": []}
    summary = {"schema": "lumo.review-response.q1-2b-summary.v2.1", "run": run_dir, "findings": F, "note": "component verification/publication only; not full-model Q1; no timing"}
    tensors_root = tensors_root or os.path.join(run_dir, "tensors")
    # ---------------- R1: binding ----------------
    lb = _load_json(os.path.join(run_dir, "LAUNCH-BINDING.json"), F, "LAUNCH-BINDING") if os.path.exists(os.path.join(run_dir, "LAUNCH-BINDING.json")) else None
    if lb is None:
        F["malformed"].append("LAUNCH-BINDING.json missing"); return summary, 2
    snaps = {k: os.path.join(run_dir, v) for k, v in (lb.get("snapshots") or {}).items()}
    for k in ("fixture_manifest", "expected_observations", "policy", "gate"):
        if k not in snaps or not os.path.exists(snaps[k]):
            F["malformed"].append(f"snapshot {k} missing"); continue
        if sha256_file(snaps[k]) != (lb.get("expect") or {}).get(k):
            F["malformed"].append(f"snapshot {k} sha differs from LAUNCH-BINDING.expect")
    if F["malformed"]:
        return summary, 2
    exp = lb["expect"]; scope = lb.get("scope") or {}; helpers = lb.get("helpers")
    missing_keys = [k for k in EXPECT_KEYS if not isinstance(exp.get(k), str) or not exp[k]]
    if missing_keys:
        F["malformed"].append(f"LAUNCH-BINDING.expect missing required source identities: {missing_keys}")
    if not isinstance(helpers, dict) or [k for k in HELPER_KEYS if not isinstance(helpers.get(k), str) or len(helpers[k]) != 64]:
        F["malformed"].append("LAUNCH-BINDING.helpers missing or incomplete (top-level, all helper keys required)")
    else:
        for hk, ek in HELPER_TO_EXPECT.items():
            if helpers[hk] != exp.get(ek):
                F["malformed"].append(f"LAUNCH-BINDING helpers[{hk}] != expect[{ek}]")
    if scope.get("block") != block or not _is_int(scope.get("repeats")) or scope["repeats"] < 2 or sorted(scope.get("processes", [])) != ["A", "B"] or not _is_int(scope.get("negatives")):
        F["malformed"].append(f"LAUNCH-BINDING scope invalid for block={block}: {scope}")
    if F["malformed"]:
        return summary, 2
    R = scope["repeats"]
    gate = _load_json(snaps["gate"], F, "gate"); fm = _load_json(snaps["fixture_manifest"], F, "fixture manifest"); obs = _load_json(snaps["expected_observations"], F, "expected observations")
    if F["malformed"]:
        return summary, 2
    # reconcile the launch binding with the UNCHANGED approved gate (the binding may not redefine or drop identities)
    rh = gate.get("reviewed_hashes") or {}; rs = gate.get("reviewed_scope") or {}
    for k in EXPECT_KEYS:
        if k == "gate":
            continue
        if rh.get(k) != exp.get(k):
            F["malformed"].append(f"LAUNCH-BINDING.expect[{k}] != gate.reviewed_hashes[{k}]")
    if str(rs.get("repeats")) != str(scope["repeats"]) or rs.get("block") != scope["block"] or str(rs.get("negatives")) != str(scope["negatives"]) or sorted(rs.get("processes", [])) != ["A", "B"]:
        F["malformed"].append(f"LAUNCH-BINDING.scope {scope} != gate.reviewed_scope {rs}")
    for k in ("policy_base_v2", "contract_v2"):
        if k not in snaps or not os.path.exists(snaps[k]) or sha256_file(snaps[k]) != exp.get(k):
            F["malformed"].append(f"snapshot {k} missing or sha != bound")
    # reducer runtime: must be the declared pinned environment (image torch, single thread, CPU-only)
    rt = reducer_runtime_identity(); summary["reducer_runtime"] = rt
    decl = lb.get("runtime") or {}
    if not decl.get("torch"):
        F["malformed"].append("LAUNCH-BINDING.runtime.torch not declared (reducer environment unbound)")
    elif rt.get("torch") != decl["torch"]:
        F["malformed"].append(f"reducer runtime torch {rt.get('torch')} != declared {decl['torch']} (reduction must run in the pinned image)")
    if rt.get("OMP_NUM_THREADS") != "1" or rt.get("MKL_NUM_THREADS") != "1" or rt.get("torch_threads") != 1:
        F["malformed"].append(f"reducer threads not pinned to 1: {rt}")
    if rt.get("cuda_available"):
        F["malformed"].append("reducer must run CPU-only (no GPU visible)")
    if F["malformed"]:
        return summary, 2
    approval = {"approved": gate.get("approved") is True, "policy_sha256": (gate.get("reviewed_hashes") or {}).get("policy"), "stage": gate.get("gate"),
                "expected_observations_sha256": (gate.get("reviewed_hashes") or {}).get("expected_observations"), "approved_run_id": gate.get("approved_run_id")}
    if approval["approved_run_id"] != lb.get("run_id") or os.path.basename(os.path.normpath(run_dir)) != lb.get("run_id"):
        F["malformed"].append("run id not bound to the approved gate")
    pol = PE21.load_policy_verified_v2_1(snaps["policy"], exp.get("policy"), approval)
    summary["policy"] = {"sha256": pol.get("_loaded_sha256"), "version": pol.get("version"), "integrity_ok": pol.get("_integrity_ok"), "approved_ok": pol.get("_approved_ok"),
                         "problems": pol.get("_load_problems"), "auth_problems": pol.get("_auth_problems")}
    if not pol.get("_approved_ok"):
        F["malformed"].append("policy v2.1 integrity/authorization failed")
    if (pol.get("base_policy_v2") or {}).get("sha256") != exp.get("policy_base_v2") or (pol.get("contract") or {}).get("sha256") != exp.get("contract_v2"):
        F["malformed"].append("policy v2.1 base-v2 / contract bindings != launch-bound base policy / contract snapshots")
    # canonical recomputation of both manifests
    def canon(doc):
        payload = {k: v for k, v in doc.items() if k not in ("generated_utc", "canonical_sha256_excluding_timestamp")}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if fm.get("schema") != "lumo.review-response.q1-2b-fixture-manifest.v1" or canon(fm) != fm.get("canonical_sha256_excluding_timestamp"):
        F["malformed"].append("fixture manifest canonical payload does not recompute")
    bound_cases, bind_problems = PE21.bind_expected_manifest(pol, obs, exp.get("expected_observations"), block) if pol.get("_integrity_ok") else ({}, ["policy integrity failed"])
    if bind_problems:
        F["malformed"].append(f"expected observations not bound: {bind_problems[:5]}")
    if F["malformed"]:
        return summary, 2
    fixtures = [e for e in fm["fixtures"] if e.get("block") == block]
    fids = [e["fixture_id"] for e in fixtures]
    if not fixtures or len(set(fids)) != len(fids) or any(not isinstance(e.get("sha256"), str) or len(e["sha256"]) != 64 or not _is_int(e.get("layers")) for e in fixtures):
        F["malformed"].append("fixture entries not typed/unique"); return summary, 2
    exp_ids = set(fids); L_by = {e["fixture_id"]: e["layers"] for e in fixtures}
    paths = None
    cases = {cid: c for cid, c in bound_cases.items() if c["fixture_id"] in exp_ids}
    per_fx = {}
    for c in cases.values():
        per_fx.setdefault(c["fixture_id"], {"output": 0, "state": 0, "paths": set()})[c["kind"]] += 1
        if c["kind"] == "state":
            per_fx[c["fixture_id"]]["paths"].add(c["path_id"])
    for fid in fids:
        pf = per_fx.get(fid)
        if pf is None or pf["output"] != L_by[fid] * 32 or pf["state"] != L_by[fid] * 28 or len(pf["paths"]) != 28:
            F["malformed"].append(f"{fid}: expected case product is not L*32 outputs + L*28 states over 28 paths ({pf})")
        paths = paths or sorted(pf["paths"]) if pf else paths
    if F["malformed"]:
        return summary, 2
    # process results
    procs = {}
    for tag in ("A", "B"):
        p = os.path.join(run_dir, f"proc{tag}", "result.json")
        if not os.path.exists(p):
            F["malformed"].append(f"missing {p}"); continue
        r = _load_json(p, F, f"proc{tag} result")
        if r is not None:
            procs[tag] = r
    if F["malformed"]:
        return summary, 2
    for tag, r in procs.items():
        if r.get("schema") != "lumo.review-response.q1-2b-process-result.v2":
            F["malformed"].append(f"proc{tag}: unsupported result schema")
        if r.get("refused_before_gpu_work") or r.get("eligible") is not True or r.get("characterization_complete") is not True or r.get("integrity_ok") is not True:
            F["malformed"].append(f"proc{tag}: refused / not eligible / not complete / integrity false")
        if r.get("block") != block or r.get("repeats") != R or not _is_int(r.get("fixtures_expected")) or r.get("fixtures_expected") != len(fids) or r.get("fixtures_done") != len(fids):
            F["malformed"].append(f"proc{tag}: block/repeats/fixture counters inconsistent with the binding")
        ids = [f.get("fixture_id") for f in r.get("fixtures", [])]
        if sorted(ids) != sorted(exp_ids):
            F["malformed"].append(f"proc{tag}: fixture set {sorted(ids)} != expected {sorted(exp_ids)}")
        att = r.get("attestation") or {}
        for k in MANDATORY_ATT:
            if k not in att or att[k] in (None, "", {}):
                F["malformed"].append(f"proc{tag}: attestation missing {k}")
        for k, bound_key in (("kernel_module_sha256", "kernel"), ("topology_sha256", "topology"), ("native_module_sha256", "native_module"), ("runner_sha256", "runner"),
                             ("oracle_sha256", "oracle"), ("fixtures_tool_sha256", "fixtures_tool"), ("image_id_arg", "image_id"), ("policy_sha256", "policy"),
                             ("expected_observations_sha256", "expected_observations"), ("fixture_manifest_sha256", "fixture_manifest")):
            if exp.get(bound_key) is not None and att.get(k) != exp.get(bound_key):
                F["malformed"].append(f"proc{tag}: attestation {k} != bound {bound_key}")
        hh = att.get("helper_hashes") or {}
        for hk in HELPER_KEYS:
            if hh.get(hk) != helpers[hk]:
                F["malformed"].append(f"proc{tag}: loaded helper {hk} hash != launch-bound helper map")
        if att.get("block") != block or att.get("repeats") != R:
            F["malformed"].append(f"proc{tag}: attestation block/repeats mismatch")
        for k, v in (lb.get("runtime") or {}).items():
            if k in ("torch", "triton", "gpu_name", "vllm") and att.get(k) != v:
                F["malformed"].append(f"proc{tag}: attestation {k}={att.get(k)!r} != declared runtime {v!r}")
    if F["malformed"]:
        return summary, 2
    a, b = procs["A"], procs["B"]
    for k in AB_RUNTIME_KEYS:
        if a["attestation"].get(k) != b["attestation"].get(k):
            F["malformed"].append(f"runtime identity {k} differs across processes")
    if F["malformed"]:
        return summary, 2
    ind = process_independence(run_dir, a["attestation"], b["attestation"])
    summary["process_independence"] = ind
    if not ind["independent"]:
        F["malformed"].append(f"processes not independent: {ind['problems']}"); return summary, 2
    # ---------------- R2: inventories, records, tensors ----------------
    inv, recs = {}, {}
    seen_tensor = set()
    for tag, r in procs.items():
        entries, inv_sha = load_inventory(run_dir, tag, F)
        if entries is None:
            continue
        ri = r.get("raw_inventory") or {}
        if ri.get("sha256") != inv_sha or ri.get("records") != len(entries):
            F["malformed"].append(f"proc{tag}: raw_inventory binding (sha/count) differs from result.json")
        inv[tag] = entries
        by = {"ref": {}, "eligibility": {}, "metrics": {}, "negative": {}}
        for e in entries:
            kind = e.get("kind"); cid = e.get("case_id")
            if kind not in by or not isinstance(cid, str):
                F["malformed"].append(f"proc{tag}: inventory entry with unknown kind/case"); continue
            if cid in by[kind]:
                F["malformed"].append(f"proc{tag}: duplicate {kind} record for {cid}"); continue
            by[kind][cid] = e
        recs[tag] = by
    if F["malformed"]:
        return summary, 2
    loaded = {"A": {}, "B": {}}
    for tag in ("A", "B"):
        by = recs[tag]
        for kind in ("ref", "eligibility", "metrics"):
            missing = [cid for cid in cases if cid not in by[kind]]
            extra = [cid for cid in by[kind] if cid not in cases]
            if missing or extra:
                F["malformed"].append(f"proc{tag}: {kind} records missing {len(missing)} / unexpected {len(extra)} (first {missing[:2] or extra[:2]})")
        if F["malformed"]:
            return summary, 2
        if block == "calibration":
            exp_neg = set()
            for fid in fids:
                L = L_by[fid]
                for l in range(L):
                    for tg in ("N1_sibling_substitution", "N4_stale_metadata", "N5_offpath_sentinels"):
                        exp_neg.add(f"NEG|{fid}|L{l}|{tg}")
                for l in (0, 1):
                    exp_neg.add(f"NEG|{fid}|L{l}|N3_ring_swap")
            got_neg = set(by["negative"].keys())
            if got_neg != exp_neg:
                F["malformed"].append(f"proc{tag}: negative record id set != exact expected product (missing {len(exp_neg - got_neg)}, extra {sorted(got_neg - exp_neg)[:2]})")
        elif by["negative"]:
            F["malformed"].append(f"proc{tag}: negative records present outside the calibration block")
        if F["malformed"]:
            return summary, 2
        for kind in ("ref", "eligibility", "metrics", "negative"):
            loaded[tag][kind] = {}
            for cid, e in by[kind].items():
                rec = verify_record(run_dir, tag, e, F, tensors_root, seen_tensor, require_tensors=(kind in ("metrics", "negative")), surface=("state" if kind == "negative" else None))
                if rec is None:
                    return summary, 2
                loaded[tag][kind][cid] = rec
    # identity fields + bindings on metrics/ref records
    for tag in ("A", "B"):
        for cid, c in cases.items():
            m = loaded[tag]["metrics"][cid]; rf = loaded[tag]["ref"][cid]; el = loaded[tag]["eligibility"][cid]
            for f in ("case_id", "stratum", "surface", "depth", "metrics_valid", "reference_valid", "candidate_sha256", "native_sha256", "c2_sha256", "candidate_dtype", "native_dtype", "heads", "execution"):
                if f not in m:
                    F["malformed"].append(f"proc{tag}: metrics {cid} missing {f}")
            if m.get("case_id") != cid or m.get("stratum") != c["stratum"] or m.get("surface") != c["surfaces"][0] or m.get("depth") != c["depths"][0] or m.get("heads") != c["heads"]:
                F["malformed"].append(f"proc{tag}: metrics {cid} identity fields != expected spec")
            if m.get("c2_sha256") != c["reference_sha256"] or rf.get("c2_sha256") != c["reference_sha256"]:
                F["malformed"].append(f"proc{tag}: {cid} C2 hash not bound to the expected reference sha")
            if m.get("native_sha256") != rf.get("native_sha256"):
                F["malformed"].append(f"proc{tag}: {cid} native tensor differs between ref and metrics records")
            ex = m.get("execution") or {}
            if ex.get("process_tag") != tag or ex.get("fixture_id") != c["fixture_id"] or ex.get("fixture_sha256") != next(e["sha256"] for e in fixtures if e["fixture_id"] == c["fixture_id"]) or ex.get("expected_reference_sha256") != c["reference_sha256"]:
                F["malformed"].append(f"proc{tag}: {cid} execution binding (process/fixture sha/reference) mismatch")
            if el.get("status") not in ("ELIGIBILITY", "MALFORMED_EVIDENCE") or el.get("case_id") != cid or el.get("policy_sha256") != pol["_loaded_sha256"]:
                F["malformed"].append(f"proc{tag}: {cid} eligibility receipt not bound")
            if m.get("candidate_dtype") not in ("torch.bfloat16", "torch.float32") or m.get("native_dtype") not in ("torch.bfloat16", "torch.float32"):
                F["malformed"].append(f"proc{tag}: {cid} dtype not in the declared set")
        if F["malformed"]:
            return summary, 2
    # native references identical across processes (never inferred from candidate)
    for cid in cases:
        if loaded["A"]["ref"][cid]["native_sha256"] != loaded["B"]["ref"][cid]["native_sha256"]:
            F["structural"].append(f"{cid}: native C1 tensor differs across processes")
        if loaded["A"]["ref"][cid].get("native_repeat_sha256s") != loaded["B"]["ref"][cid].get("native_repeat_sha256s"):
            F["structural"].append(f"{cid}: native repeat census differs across processes")
    # ---------------- R3: repeats/determinism from raw hashes ----------------
    fa = {f["fixture_id"]: f for f in a["fixtures"]}; fb = {f["fixture_id"]: f for f in b["fixtures"]}
    rows = []
    for fid in sorted(exp_ids):
        L = L_by[fid]
        for t, r in (("A", fa[fid]), ("B", fb[fid])):
            if r.get("status") != "DONE":
                F["structural"].append(f"{fid} proc{t}: status {r.get('status')}"); continue
            st = r.get("structural") or {}
            for k in STRUCT_KEYS:
                if st.get(k) is not True:
                    F["structural"].append(f"{fid} proc{t}: {k} = {st.get(k)}")
            ph = r.get("phase") or {}
            if not (isinstance(ph.get("R_sealed_utc"), str) and isinstance(ph.get("C_first_scan_utc"), str) and ph["R_sealed_utc"] <= ph["C_first_scan_utc"]):
                F["structural"].append(f"{fid} proc{t}: reference phase not derivably before the first candidate scan ({ph})")
            inv_f = [e for e in inv[t] if e.get("fixture_id") == fid]
            ref_max = max((e["utc"] for e in inv_f if e["kind"] in ("ref", "eligibility")), default=None); met_min = min((e["utc"] for e in inv_f if e["kind"] == "metrics"), default=None)
            if ref_max is None or met_min is None or ref_max > met_min:
                F["structural"].append(f"{fid} proc{t}: inventory timestamps do not show reference records sealed before candidate metrics")
            idx_ids = [c.get("case_id") for c in (r.get("cases") or [])]
            if sorted(idx_ids) != sorted(cid for cid in cases if cases[cid]["fixture_id"] == fid) or len(set(idx_ids)) != len(idx_ids):
                F["structural"].append(f"{fid} proc{t}: fixture case index does not reconcile with the expected metrics set")
            reps = r.get("repeat_hashes") or []
            if [x.get("repeat") for x in reps] != list(range(R)) or any(isinstance(x.get("repeat"), bool) for x in reps):
                F["structural"].append(f"{fid} proc{t}: repeat indices != {list(range(R))}")
            for x in reps:
                if x.get("ring_bytes_exact") is not True or not isinstance(x.get("scan_out_sha256"), list) or len(x["scan_out_sha256"]) != L or not isinstance(x.get("scan_node_sha256"), list) or len(x["scan_node_sha256"]) != L:
                    F["structural"].append(f"{fid} proc{t} rep{x.get('repeat')}: ring flag / scan hash product invalid")
                pub = x.get("publish") or {}
                if sorted(pub.keys()) != sorted(paths):
                    F["structural"].append(f"{fid} proc{t} rep{x.get('repeat')}: publish map paths != expected 28 paths")
                for pid, v in pub.items():
                    if not (_is_int(v.get("replays_enqueued")) and v.get("replays_enqueued") == 1 and v.get("pointer_identity_unchanged") is True and v.get("control_rows_intact") is True and v.get("staging_neutral_tail") is True
                            and isinstance(v.get("state_sha256"), list) and len(v["state_sha256"]) == L and all(isinstance(s, str) and len(s) == 64 for s in v["state_sha256"])):
                        F["structural"].append(f"{fid} proc{t} rep{x.get('repeat')} {pid}: per-publication evidence invalid")
            # within-process determinism recomputed from hashes
            if reps and any(x.get("scan_out_sha256") != reps[0].get("scan_out_sha256") or x.get("scan_node_sha256") != reps[0].get("scan_node_sha256")
                            or {k: v.get("state_sha256") for k, v in (x.get("publish") or {}).items()} != {k: v.get("state_sha256") for k, v in (reps[0].get("publish") or {}).items()} for x in reps):
                F["structural"].append(f"{fid} proc{t}: within-process candidate hashes differ across repeats")
            nat = r.get("native_repeats") or []
            if [x.get("repeat") for x in nat] != list(range(R)) or any(x.get("state_sha256") != nat[0].get("state_sha256") or x.get("out_sha256") != nat[0].get("out_sha256") for x in nat) \
               or any(len(x.get("state_sha256") or []) != L or any(len(d) != 32 for d in x.get("state_sha256") or []) for x in nat):
                F["structural"].append(f"{fid} proc{t}: native repeat census incomplete or non-identical")
        ra, rb = fa[fid], fb[fid]
        if ra.get("status") == "DONE" and rb.get("status") == "DONE":
            for ha, hb in zip(ra["repeat_hashes"], rb["repeat_hashes"]):
                if ha.get("scan_out_sha256") != hb.get("scan_out_sha256") or ha.get("scan_node_sha256") != hb.get("scan_node_sha256"):
                    F["structural"].append(f"{fid} rep{ha.get('repeat')}: scan outputs differ across processes")
                for pid in paths:
                    if (ha.get("publish") or {}).get(pid, {}).get("state_sha256") != (hb.get("publish") or {}).get(pid, {}).get("state_sha256"):
                        F["structural"].append(f"{fid} rep{ha.get('repeat')} {pid}: published states differ across processes")
            if (ra.get("native_repeats") or [{}])[0].get("state_sha256") != (rb.get("native_repeats") or [{}])[0].get("state_sha256"):
                F["structural"].append(f"{fid}: native C1 census differs across processes")
        # metrics records' candidate hashes must equal the repeat-0 raw hashes (records are not free-standing claims)
        for t in ("A", "B"):
            r = fa[fid] if t == "A" else fb[fid]
            if r.get("status") != "DONE":
                continue
            r0 = r["repeat_hashes"][0]
            for cid, c in cases.items():
                if c["fixture_id"] != fid:
                    continue
                m = loaded[t]["metrics"][cid]; l = c["instance"]
                def _at(lst, i):
                    return lst[i] if isinstance(lst, list) and 0 <= i < len(lst) else None
                if c["kind"] == "output":
                    want = (_at(r0.get("scan_node_sha256"), l) or {}).get(str(c["node"]))
                else:
                    want = _at(((r0.get("publish") or {}).get(c["path_id"]) or {}).get("state_sha256"), l)
                if want is None or m.get("candidate_sha256") != want:
                    F["structural"].append(f"{fid} proc{t}: {cid} metrics candidate hash != repeat-0 raw hash")
                nat0 = (r.get("native_repeats") or [{}])[0]
                nl = _at(nat0.get("out_sha256") if c["kind"] == "output" else nat0.get("state_sha256"), l)
                nwant = (nl or {}).get(str(c["node"] if c["kind"] == "output" else c["accepted_nodes"][-1]))
                if nwant is None or m.get("native_sha256") != nwant:
                    F["structural"].append(f"{fid} proc{t}: {cid} metrics native hash != native census")
        rows.append({"fixture_id": fid, "findings": ra.get("findings"), "cases": len([1 for c in cases.values() if c["fixture_id"] == fid])})
        # ---------------- R4: negatives, both processes, exact product ----------------
        if block == "calibration":
            for t, r in (("A", ra), ("B", rb)):
                negs = r.get("negatives") or []
                tags = [n.get("tag") for n in negs]
                if sorted(tags) != sorted(NEG_TAGS):
                    F["structural"].append(f"{fid} proc{t}: negative tag set {sorted(tags)} != {sorted(NEG_TAGS)}"); continue
                byt = {n["tag"]: n for n in negs}
                expect_n = {"N1_sibling_substitution": L, "N3_ring_swap": 2, "N4_stale_metadata": L, "N5_offpath_sentinels": L}
                for tag, n_exp in expect_n.items():
                    n = byt[tag]
                    if not (_is_int(n.get("records_expected")) and _is_int(n.get("records_sealed")) and n["records_expected"] == n["records_sealed"] == n_exp):
                        F["structural"].append(f"{fid} proc{t}: {tag} record product {n.get('records_sealed')}/{n.get('records_expected')} != {n_exp}")
                    inst = range(L) if tag != "N3_ring_swap" else (0, 1)
                    for l in inst:
                        g = loaded[t]["negative"].get(f"NEG|{fid}|L{l}|{tag}")
                        if g is None:
                            F["structural"].append(f"{fid} proc{t}: {tag} L{l} sealed negative record missing"); continue
                        tgt = cases.get(f"{fid}|L{l}|state|n14"); ex = g.get("execution") or {}
                        if (g.get("negative") != tag or g.get("surface") != "state" or g.get("depth") != 5 or g.get("heads") != 48 or ex.get("instance") != l or ex.get("negative") != tag
                                or ex.get("path_id") != "n14" or tgt is None or g.get("c2_sha256") != tgt["reference_sha256"] or g.get("native_sha256") != loaded[t]["ref"][tgt["case_id"]]["native_sha256"]):
                            F["structural"].append(f"{fid} proc{t}: {tag} L{l} negative record identity/target binding invalid")
                n5 = byt["N5_offpath_sentinels"]
                if not _is_int(n5.get("leaked_instances")) or n5["leaked_instances"] != 0:
                    F["structural"].append(f"{fid} proc{t}: N5 leaked_instances = {n5.get('leaked_instances')!r} (need int 0)")
                for l in range(L):
                    g = loaded[t]["negative"].get(f"NEG|{fid}|L{l}|N5_offpath_sentinels")
                    clean_list = ((r.get("repeat_hashes") or [{}])[0].get("publish") or {}).get("n14", {}).get("state_sha256")
                    clean_l = clean_list[l] if isinstance(clean_list, list) and l < len(clean_list) else None
                    if g is not None and not (g.get("poisoned_equals_clean_candidate") is True and g.get("poisoned_candidate_sha256") == g.get("clean_candidate_sha256") == g.get("candidate_sha256")
                                              and clean_l is not None and g.get("clean_candidate_sha256") == clean_l):
                        F["structural"].append(f"{fid} proc{t} L{l}: N5 poisoned C0 != clean C0 (causal off-path leak or unbound hashes)")
                n7 = byt["N7_incumbent_twice"]
                if n7.get("harness_rejects") is not True or not _is_int(n7.get("replays_enqueued")) or n7["replays_enqueued"] != 0:
                    F["structural"].append(f"{fid} proc{t}: N7 incumbent-twice not rejected")
    if F["structural"]:
        summary["rows"] = rows; return summary, 5
    # ---------------- N: numerical (candidate-blind eligibility recomputed; both processes) ----------------
    case_results = {"A": [], "B": []}
    recomputed = 0
    fx_cache = {}
    for tag in ("A", "B"):
        for cid, c in cases.items():
            rf = loaded[tag]["ref"][cid]; m = loaded[tag]["metrics"][cid]
            el = PE21.reference_only_eligibility(pol, c, [rf], bound_cases=cases, fixture_sha256=next(e["sha256"] for e in fixtures if e["fixture_id"] == c["fixture_id"]))
            sealed_el = loaded[tag]["eligibility"][cid]
            if el.get("evidence_sha256") != sealed_el.get("evidence_sha256") or el.get("eligible") != sealed_el.get("eligible"):
                F["malformed"].append(f"proc{tag}: {cid} sealed eligibility receipt does not recompute"); continue
            if recompute == "all":
                ok, why = _recompute_metrics(run_dir, tensors_root, m, c, fixtures, fx_cache)
                if not ok:
                    F["malformed"].append(f"proc{tag}: {cid} metric arrays do not recompute from retained tensors: {why}"); continue
                recomputed += 1
            case_results[tag].append(PE21.evaluate_case_v2_1(pol, c, [m], el, bound_cases=cases))
    if F["malformed"]:
        summary["rows"] = rows; return summary, 2
    agg = {t: PE21.aggregate_v2_1(list(cases.keys()), case_results[t]) for t in ("A", "B")}
    verdicts = {t: {} for t in ("A", "B")}
    for t in ("A", "B"):
        for r in case_results[t]:
            key = r.get("verdict") or r.get("status"); verdicts[t][key] = verdicts[t].get(key, 0) + 1
    # negative power (both processes): N1/N3/N4 must FAIL the paired rules vs the correct references; N5 paired result reported only.
    # Every negative's six arrays are first RECOMPUTED from its retained tensors + regenerated C2 (never trusted as declared).
    neg_power = {"A": {}, "B": {}}
    for t in ("A", "B"):
        for cid, g in loaded[t]["negative"].items():
            parts = cid.split("|"); fid = parts[1]; l = int(parts[2][1:]); tag = parts[3]
            target_cid = f"{fid}|L{l}|state|n14"; c = cases.get(target_cid)
            if c is None or g.get("negative") != tag:
                F["structural"].append(f"proc{t}: negative {cid} has no target case / tag mismatch"); continue
            if recompute == "all":
                ok, why = _recompute_metrics(run_dir, tensors_root, g, c, fixtures, fx_cache)
                if not ok:
                    F["malformed"].append(f"proc{t}: negative {cid} arrays do not recompute from retained tensors: {why}"); continue
                recomputed += 1
            rf = loaded[t]["ref"][target_cid]
            el = PE21.reference_only_eligibility(pol, c, [rf], bound_cases=cases)
            gg = dict(g); gg["case_id"] = target_cid   # evaluate the poisoned/corrupted publication against the CORRECT n14 references
            res = PE21.evaluate_case_v2_1(pol, c, [gg], el, bound_cases=cases)
            neg_power[t].setdefault(tag, []).append(res.get("verdict") or res.get("status"))
    if F["malformed"]:
        summary["rows"] = rows; return summary, 2
    # negative POWER enforcement, explicitly for BOTH processes: every N1/N3/N4 record (recomputed above) must FAIL the paired rules
    expected_neg_counts = {"N1_sibling_substitution": sum(L_by.values()), "N3_ring_swap": 2 * len(fids), "N4_stale_metadata": sum(L_by.values())}
    for t in ("A", "B"):
        for tag, n_exp in expected_neg_counts.items():
            got = neg_power[t].get(tag) or []
            if len(got) != n_exp or any(v != "FAIL" for v in got):
                F["structural"].append(f"proc{t}: negative {tag} power not established ({len(got)}/{n_exp} records, verdicts {sorted(set(got))}); a no-op negative is not a control")
        n5 = neg_power[t].get("N5_offpath_sentinels") or []
        summary.setdefault("n5_paired_diagnostic", {})[t] = {v: n5.count(v) for v in set(n5)}   # diagnostic only; N5's requirement is the causal poisoned==clean invariant above
    summary.update({"rows": rows, "cases_expected": len(cases), "cases_evaluated": {t: len(case_results[t]) for t in ("A", "B")}, "case_verdicts": verdicts, "aggregate": agg,
                    "negative_power": neg_power, "recomputed_records": recomputed, "process_independence": summary.get("process_independence"), "attestation_A": a["attestation"], "attestation_B": b["attestation"], "tensors_verified": len(seen_tensor)})
    if F["structural"]:
        return summary, 5
    for t in ("A", "B"):
        if agg[t]["aggregate"] == "REFUSED_OR_MALFORMED":
            F["malformed"].append(f"proc{t}: {agg[t]}"); return summary, 2
    if any(agg[t]["aggregate"] == "FAIL" for t in ("A", "B")):
        F["numerical"].append("paired non-regression violated in at least one case"); return summary, 6
    if any(agg[t]["aggregate"] != "PASS" for t in ("A", "B")):
        F["uncovered"].append("not every expected observation is covered in both processes"); return summary, 4
    return summary, 0


def _recompute_metrics(run_dir, tensors_root, m, c, fixtures, fx_cache):
    """Recompute the six metric arrays from the retained candidate/native tensors and a CPU-regenerated C2."""
    try:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import q1_2b_fixtures as FX
    except Exception as e:  # noqa: BLE001
        return False, f"torch/fixtures unavailable ({type(e).__name__})"
    def load(ref):
        p = os.path.join(tensors_root, ref["sha256"] + ".bin"); b = open(p, "rb").read()
        dt = {"torch.bfloat16": torch.int16, "torch.float32": torch.float32}[ref["dtype"]]
        t = torch.frombuffer(bytearray(b), dtype=dt).reshape(ref["shape"])
        return t.view(torch.bfloat16) if ref["dtype"] == "torch.bfloat16" else t
    Y = load(m["candidate_tensor"]); N_ = load(m["native_tensor"])
    fid = c["fixture_id"]
    if fid not in fx_cache:
        e = next(x for x in fixtures if x["fixture_id"] == fid)
        p = os.path.join(run_dir, "fixtures", e["path"])
        if not os.path.exists(p) or hashlib.sha256(open(p, "rb").read()).hexdigest() != e["sha256"]:
            return False, "fixture .pt not available/verified for C2 regeneration"
        fx = torch.load(p, map_location="cpu", weights_only=False); fx_cache[fid] = {"fx": fx, "c2": {}}
    fx = fx_cache[fid]["fx"]; l = c["instance"]
    if l not in fx_cache[fid]["c2"]:
        fx_cache[fid]["c2"][l] = FX.c2_node_refs(fx["instances"][l], float(fx["meta"]["scale"]))
    st, ou = fx_cache[fid]["c2"][l]
    node = c["node"] if c["kind"] == "output" else c["accepted_nodes"][-1]
    R = ou[node] if c["kind"] == "output" else st[node]
    if hashlib.sha256(R.numpy().tobytes()).hexdigest() != c["reference_sha256"]:
        return False, "regenerated C2 hash != expected reference sha"
    Y64, N64, R64 = Y.to(torch.float64), N_.to(torch.float64), R.to(torch.float64)
    def per_head(d):
        f = d.flatten(1); return torch.sqrt((f * f).mean(dim=1)).tolist(), f.abs().max(dim=1).values.tolist()
    rms0, mx0 = per_head(Y64 - R64); rms1, mx1 = per_head(N64 - R64); rms2, mx2 = per_head(R64)
    got = {"rms_C0_C2": rms0, "maxabs_C0_C2": mx0, "rms_C1_C2": rms1, "maxabs_C1_C2": mx1, "rms_C2": rms2, "maxabs_C2": mx2}
    for k, v in got.items():
        if list(m.get(k, [])) != v:
            return False, f"{k} differs"
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--block", default="calibration"); ap.add_argument("--recompute", default="all", choices=["all", "none"])
    ap.add_argument("--tensors", default=None)
    a = ap.parse_args()
    summary, code = reduce_run(a.run, a.block, a.recompute, a.tensors)
    with open(os.path.join(a.run, "summary.v2.json"), "x") as f:
        json.dump(summary, f, indent=1, allow_nan=False, default=str)
    print(json.dumps({"exit": code, "findings": {k: len(v) for k, v in summary["findings"].items()}, "aggregate": summary.get("aggregate"), "case_verdicts": summary.get("case_verdicts")}, default=str))
    sys.exit(code)


if __name__ == "__main__":
    main()
