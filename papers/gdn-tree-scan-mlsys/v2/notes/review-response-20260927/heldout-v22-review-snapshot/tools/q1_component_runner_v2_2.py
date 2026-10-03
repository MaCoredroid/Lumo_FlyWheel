#!/usr/bin/env python3
"""Q1.2b candidate GDN component runner **v2.2** (runs INSIDE the pinned image, GPU; case loop CPU-testable via a backend).

v2.2 = v2.1 + explicit held-out block support (parent instruction 2026-09-28T01:09:47Z).  The v2.1 guard hard-coded
`eligible = block == "calibration"`; v2.2 admits exactly ONE other block, the manifest/policy-domain held-out block
"evaluation", and only when a numerical gate (gate "Q1.2b") names every held-out fixture id, reviewed_scope.block is
"evaluation", negatives are 0 (negative mutations stay calibration-only), --limit is 0 (complete coverage), the gate names
the accepted calibration run that established negative power, and reviewed_hashes.runner binds THIS file's bytes.  Any other
block string is refused before GPU work.  No evaluator/policy/constant/fixture/kernel change; lifecycle setup, case loop and
attestation are the v2.1 bytes (see runner_v2_1_to_v2_2.diff).

v2.1 = v2 + the ONE lifecycle-setup repair required after the failed approved run q12b-calibration-20260928T002602Z
(process A died in the production boot warm: "FR13 fixed32 postprocess boot warm has no clean conv lease").  The v2
Backend allocated 48 independent SSM banks and skipped the production convolution lease.  v2.1 builds the persistent
component storage the way the deployed route does and runs the PUBLIC production boot sequence unchanged:

  16 page-shared raw buffers (deployed page geometry: 2,097,152 bf16 elements per row) each viewed as ONE conv bank
  (rows, 10240, 34) bf16 [channel-contiguous SD layout] and ONE SSM bank (rows, 48, 128, 128) fp32 on the SAME untyped
  storage -> 16 exact alias classes of width 3 ({a, a+16, a+32}), each spanning the three builder SSI groups; the three
  layers of a class use DISTINCT running/scratch/control rows (per alias rank);
  register_fixed32_conv_col0_ssi_group x3 -> preseed_fixed32_committer_graphs_all_batches -> preseed_fixed32_conv_col0_pregather
  (commit SSI (48,B,32) int32, accepted_paths (B,16)/lens (B,) int32 contiguous, 48 distinct bf16 source stagings
  (B*36, 10240), int64 state_src of length 32*34 from fr13_tree_conv_fused.build_tree_conv_state_src_indices(width=4))
  -> selfcheck_fixed32_conv_col0_ssi_sources -> audit_fixed32_conv_commit_lease -> warm_fixed32_committer_graphs_all_batches;
  the warm's restoration flags and the zero measured counters are asserted and attested.  No guard is bypassed and no
  private lease dictionary is filled by hand.  This is setup fidelity only: convolution remains OUT OF SCOPE of Q1.2b.

v2 text follows (unchanged apart from the Backend, the helper list and the attestation additions).


v2 repairs (parent runner review 2026-09-27 + reducer review R1-R4 + policy reviewer):
  * PHASE R (reference-first, candidate-blind): C2 recompute + hash check, then C1 native chain for EVERY node and
    EVERY repeat on an independent scratch bank BEFORE any candidate scan; native tensors retained content-addressed;
    candidate-blind reference records (`ref/`) sealed and reference-only eligibility (evaluator v2.1, Q1.2b domain
    bound prospectively) computed BEFORE the first candidate launch; the first candidate scan timestamp is recorded.
  * PHASE C: candidate scan/publish per repeat; candidate tensors retained content-addressed (dedup across equal
    repeats/processes); metric records carry candidate/native/C2 hashes + tensor refs + execution binding and are
    sealed with a record sha; every record is appended to a streaming per-process raw inventory (JSONL) which the
    terminal result.json binds (path, case identity, bytes, sha) -- omission/duplication is detectable.
  * Native repeat census: C1 state/out hashes for every (instance, node) and every repeat, within-process equality
    verified explicitly; reducer verifies cross-process.
  * N5 = causal off-path invariance: poisoned C0 publication must equal the SAME unpoisoned C0 publication bitwise
    (hash + retained bytes); the paired C1/C2 record of the poisoned publication is sealed separately as diagnostic.
  * Exact negative product per fixture/process: N1 L records, N3 2 records (instances 0,1), N4 L, N5 L, N7 counter.
  * Receipts: actual executed scan route stamp (kernel subtree cache `last_executed_gdn`), expected schedule, tree
    profile/valid mask/geometry override/scan_align, committer options, native dispatch identity, every loaded
    campaign helper hash, projected archive bytes.  `bank_off16` is a LEGACY argument: fixed32 replay binds the
    preseeded bank tuple and never reads it as a pointer table (documented, not described as one).
Exit: 0 complete + integrity ok; 2 refusal before GPU; 3 integrity failure (evidence retained).
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, socket, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import q1_oracle as O  # noqa: E402
import q1_2b_fixtures as FX  # noqa: E402
import q1_policy_evaluator_v2_1 as PE21  # noqa: E402

SCHEMA_RESULT = "lumo.review-response.q1-2b-process-result.v2"   # unchanged schema: only the setup changed
EXPECTED_KERNEL_SHA = "d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8"
EXPECTED_TOPOLOGY_SHA = "c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc"
EXPECTED_NATIVE_SHA = "000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f"
REQUIRED_ENV = {"FR13_FIXED32_MODE": "hydra27_fixed32", "FR13_SUBTREE_PARALLEL": "1", "FR13_SCAN_ALIGN": "0", "FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8",
                "FR13_RING_EXPORT": "1", "FR13_TREE_RUNROW_INIT": "1", "FR13_FLAGS_INKERNEL": "1", "FR13_FIXED32_COMMIT_DEVICE_FILL": "1",
                "FR13_FIXED32_KV_REMAP16": "1", "FR13_FIXED32_COMMITTER_LAYER_BATCH": "0", "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION": "0",
                "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": "0", "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE": "0", "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL": "0"}
HELPER_FILES = ("q1_component_runner_v2_1.py", "q1_oracle.py", "q1_2b_fixtures.py", "q1_policy_evaluator.py", "q1_policy_evaluator_v2.py", "q1_policy_evaluator_v2_1.py", "q1_bf16_ulp.py")
REPO_HELPER_FILES = ("scripts/fr13_fixed32_topology.py", "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py", "src/lumo_flywheel_serving/fr13_tree_conv_fused.py")
H, HV, K, V, N = FX.H, FX.HV, FX.K, FX.V, FX.N_PHYS
# ---- deployed page geometry (production GDN cache page) and per-alias-rank row assignment -------------------------
CONV_C, CONV_L = 10240, 34                 # conv channels (2*16*128 + 48*128), conv state length (kernel-1 + 31 spec)
PAGE_ELEMS_BF16 = 2_097_152                 # deployed page: 4 MiB per row
SOURCE_ROWS = 36                            # conv commit source rows per batch (production contract)
CAPACITY = 1                                # server capacity of this component harness (B=1)
ALIAS_CLASSES, ALIAS_WIDTH, SSI_GROUPS = 16, 3, 3
ROWS = 24           # bank rows per page: 0 null; 1..3 warm alias rows (kernel-owned, restored); 4..6 running rows by alias rank;
                    # 7..9 scratch rows by rank; 10..15 control rows (2 per rank); 16..21 spare; 22 committer-preseed safe root; 23 top
WARM_ROWS = (1, 2, 3)
def alias_rank(l): return l // 16
def alias_class(l): return l % 16
def run_row(l): return 4 + alias_rank(l)
def scratch_row(l): return 7 + alias_rank(l)
def control_rows(l): return (10 + 2 * alias_rank(l), 11 + 2 * alias_rank(l))
def layer_name(l): return f"L{l:02d}"
N5_TARGET_PATH = "n14"      # spine depth 4: nodes [0,1,4,9,14]
N1_SIBLING_PATH = [0, 1, 4, 9, 15]
N3_INSTANCES = (0, 1)
N4_PREVIOUS_PATH = [0, 1, 4]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_bytes(t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return t.numpy().tobytes()


def sha_t(t):
    return hashlib.sha256(tensor_bytes(t)).hexdigest()


def nonfinite(t):
    return int((~torch.isfinite(t.float())).sum().item())


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True).encode()).hexdigest()


def masks_from_parent(parent, device):
    n = len(parent); strict = torch.zeros((n, n), dtype=torch.int32); visible = torch.zeros((n, n), dtype=torch.int32)
    for i in range(n):
        visible[i, i] = 1; cur = parent[i]
        while cur >= 0:
            strict[i, cur] = 1; visible[i, cur] = 1; cur = parent[cur]
    return strict.to(device), visible.to(device)


def boot_lifecycle(kernel, be):
    """PUBLIC production boot sequence on the backend's persistent tensors (object identities preserved; nothing private touched):
    subtree_preseed -> register_fixed32_conv_col0_ssi_group x3 -> preseed_fixed32_committer_graphs_all_batches -> preseed_fixed32_conv_col0_pregather
    -> selfcheck_fixed32_conv_col0_ssi_sources (per B) -> audit_fixed32_conv_commit_lease -> warm_fixed32_committer_graphs_all_batches;
    then the warm receipt (ready + every restoration flag) and zero measured counters are REQUIRED.  Raises on any failure."""
    boot = {"sequence": []}
    kernel.subtree_preseed(kernel._FR13_FIXED32_PARENT, N, HV, V, K, be.device); boot["sequence"].append("subtree_preseed")
    for g in range(SSI_GROUPS):
        kernel.register_fixed32_conv_col0_ssi_group(layer_names=be.layer_order[16 * g:16 * (g + 1)], spec_state_indices=be.ssi_groups[g], max_batch_size=CAPACITY)
    boot["sequence"].append("register_fixed32_conv_col0_ssi_group x3")
    preseed = kernel.preseed_fixed32_committer_graphs_all_batches(
        banks_list=be.banks, spec_state_indices=be.spec_idx, accepted_paths=be.accepted_paths, accepted_lens=be.accepted_lens,
        k_rings=be.ring_k, k_norm_rings=None, gate_rings=None, v_rings=be.ring_v, a_rings=be.ring_a, b_rings=be.ring_b,
        A_logs=be.A_logs, dt_biases=be.dt_biases, num_layers=be.L, max_batch_size=CAPACITY, output_scale=be.scale,
        use_qk_l2norm_in_kernel=True, burn_node_bank=False, root_node=0, max_path=16); boot["sequence"].append("preseed_fixed32_committer_graphs_all_batches")
    conv_contract = kernel.preseed_fixed32_conv_col0_pregather(
        conv_banks=be.conv_banks, ssm_banks=be.banks, layer_order=be.layer_order, max_batch_size=CAPACITY, commit_spec_state_indices=be.spec_idx,
        accepted_paths=be.accepted_paths, accepted_lens=be.accepted_lens, commit_source_stagings=be.source_stagings, commit_state_src=be.state_src,
        source_rows_per_batch=SOURCE_ROWS); boot["sequence"].append("preseed_fixed32_conv_col0_pregather")
    for b in range(1, CAPACITY + 1):
        kernel.selfcheck_fixed32_conv_col0_ssi_sources(num_spec_decodes=b)
    boot["sequence"].append("selfcheck_fixed32_conv_col0_ssi_sources")
    conv_audit = kernel.audit_fixed32_conv_commit_lease(); boot["sequence"].append("audit_fixed32_conv_commit_lease")
    warm = kernel.warm_fixed32_committer_graphs_all_batches(); boot["sequence"].append("warm_fixed32_committer_graphs_all_batches")
    warm_counters = kernel.fixed32_committer_warmup_counters()
    after = kernel.fixed32_committer_counters()
    flags = ("bank_state_restored", "conv_bank_state_restored", "conv_staging_state_restored", "input_state_restored", "measured_state_restored", "route_lease_current")
    boot["warm_ready"] = bool(warm_counters.get("ready")) and warm_counters.get("classification") == "unmeasured_boot"
    boot["restoration_flags"] = {k: bool(warm_counters.get(k)) for k in flags}
    boot["measured_counters_zero"] = int(after["actual_replays_enqueued"]) == 0 and all(int(v) == 0 for v in dict(after.get("actual_replays_by_batch", {})).values())
    boot["boundary"] = "conv warm/setup executed through the public production lifecycle; NO conv qualification cases; convolution numerics remain out of Q1.2b scope"
    if not (boot["warm_ready"] and all(boot["restoration_flags"].values()) and boot["measured_counters_zero"]):
        raise RuntimeError(f"fixed32 boot warm did not leave a clean measured state: {boot}")
    return boot, preseed, conv_contract, conv_audit, warm, warm_counters


def build_alias_pages(device, rows):
    """Deployed-layout component storage: 16 raw pages; per page ONE conv view (rows, CONV_C, CONV_L) bf16 [SD: as_strided (rows, L, C) then
    transpose -> inner strides (1, C)] and ONE SSM view (rows, HV, V, K) fp32 at the byte offset right after the conv row region, on the SAME
    untyped storage.  Layer l uses page l % 16, so the 48-tuple has 16 exact alias classes of width 3 spanning the three 16-layer SSI groups.
    Returns identity-preserving tuples (the SAME view objects are reused for the three layers of a class)."""
    conv_elems = CONV_C * CONV_L
    assert (conv_elems * 2) % 16 == 0 and conv_elems * 2 + HV * V * K * 4 <= PAGE_ELEMS_BF16 * 2, "page cannot hold conv + ssm rows"
    raws, conv_views, ssm_views = [], [], []
    for a in range(ALIAS_CLASSES):
        raw = torch.zeros((rows * PAGE_ELEMS_BF16,), dtype=torch.bfloat16, device=device)
        conv = torch.as_strided(raw, size=(rows, CONV_L, CONV_C), stride=(PAGE_ELEMS_BF16, CONV_C, 1)).transpose(1, 2)
        raw32 = raw.view(torch.float32)
        ssm = torch.as_strided(raw32, size=(rows, HV, V, K), stride=(PAGE_ELEMS_BF16 // 2, V * K, K, 1), storage_offset=conv_elems // 2)
        raws.append(raw); conv_views.append(conv); ssm_views.append(ssm)
    conv_banks = tuple(conv_views[alias_class(l)] for l in range(48)); ssm_banks = tuple(ssm_views[alias_class(l)] for l in range(48))
    geometry = {"page_elems_bf16": PAGE_ELEMS_BF16, "rows": rows, "conv_shape": list(conv_banks[0].shape), "conv_stride": list(conv_banks[0].stride()),
                "ssm_shape": list(ssm_banks[0].shape), "ssm_stride": list(ssm_banks[0].stride()), "ssm_storage_offset_fp32": conv_elems // 2,
                "alias_map": [[a, a + 16, a + 32] for a in range(ALIAS_CLASSES)]}
    return {"raws": tuple(raws), "conv_banks": conv_banks, "ssm_banks": ssm_banks, "geometry": geometry}


def validate_alias_topology(conv_banks, ssm_banks, groups_of, capacity=CAPACITY):
    """CPU re-statement of the kernel's alias/pointer contract (kernel _validate_fixed32_conv_pregather_preseed) on the given bank tuples.
    Returns a list of problems (empty = ok).  Row POLICY is checked separately by validate_row_policy (module constants)."""
    pr = []
    if len(conv_banks) != 48 or len(ssm_banks) != 48:
        return ["need exactly 48 conv banks and 48 SSM banks"]
    a0 = conv_banks[0]; shape, stride = tuple(a0.shape), tuple(a0.stride())
    if len(shape) != 3 or shape[0] < capacity or (stride[1], stride[2]) not in ((shape[2], 1), (1, shape[1])):
        pr.append(f"conv anchor not page-safe: shape={shape} stride={stride}")
    span = (shape[1] - 1) * stride[1] + (shape[2] - 1) * stride[2] + 1 if len(shape) == 3 else None
    if span is not None and stride[0] < span:
        pr.append("conv row stride smaller than the logical row span")
    by_ptr = {}
    for i, (cb, sb) in enumerate(zip(conv_banks, ssm_banks)):
        if tuple(cb.shape) != shape or tuple(cb.stride()) != stride or cb.dtype != a0.dtype or cb.data_ptr() % 16:
            pr.append(f"conv bank {i} contract/alignment mismatch")
        if tuple(sb.shape) != (shape[0], HV, V, K) or sb.dtype != torch.float32:
            pr.append(f"SSM bank {i} is not the reviewed fp32 (rows,48,128,128) state")
        if cb.untyped_storage().data_ptr() != sb.untyped_storage().data_ptr():
            pr.append(f"conv/SSM pair {i} not on the same storage")
        by_ptr.setdefault(cb.data_ptr(), []).append(i)
    classes = sorted(by_ptr.values(), key=lambda x: x[0])
    if len(classes) != ALIAS_CLASSES or any(len(c) != ALIAS_WIDTH for c in classes) or any(len({groups_of(i) for i in c}) != SSI_GROUPS for c in classes):
        pr.append(f"conv alias topology is not 16 exact aliases of width 3 spanning 3 SSI groups ({len(classes)} classes)")
    sby = {}
    for i, sb in enumerate(ssm_banks):
        sby.setdefault(sb.data_ptr(), []).append(i)
    if sorted(sby.values(), key=lambda x: x[0]) != classes:
        pr.append("conv/SSM alias topology mismatch")
    if any(int(b.shape[0]) <= 3 * capacity for b in (*conv_banks, *ssm_banks)):
        pr.append("banks have no isolated alias rows for the warm (need rows > 3*capacity)")
    if int(shape[0]) < capacity + 1:
        pr.append("too few bank rows for the committer preseed safe roots")
    return pr


def validate_row_policy(rows=None, capacity=CAPACITY):
    """The harness's row assignment on page-shared banks: per alias class distinct running/scratch/control rows, none on the kernel-owned
    warm rows (1..3*capacity), all below the committer-preseed safe roots (rows-capacity-1 .. rows-2)."""
    rows = ROWS if rows is None else rows
    pr = []
    warm = set(range(1, 3 * capacity + 1))
    for a in range(ALIAS_CLASSES):
        c = [a, a + 16, a + 32]
        run, scr, ctl = [run_row(i) for i in c], [scratch_row(i) for i in c], [r for i in c for r in control_rows(i)]
        if len(set(run)) != len(c) or len(set(scr)) != len(c) or len(set(ctl)) != 2 * len(c) or (set(run) & set(scr)) or (set(run) | set(scr)) & set(ctl):
            pr.append(f"alias class {c}: rows not distinct per rank")
        used = set(run) | set(scr) | set(ctl)
        if 0 in used or (used & warm) or max(used) >= rows - capacity - 1:
            pr.append(f"alias class {c}: row assignment collides with row 0, the warm rows {sorted(warm)} or the safe roots (rows={rows})")
    return pr


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if not torch.is_tensor(v)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj if not torch.is_tensor(v)]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


class TensorStore:
    """Content-addressed, write-once tensor archive shared by both processes (dedup across equal repeats/processes)."""

    def __init__(self, root):
        self.root = root; os.makedirs(root, exist_ok=True); self.written = 0; self.reused = 0; self.bytes_written = 0

    def put(self, t):
        b = tensor_bytes(t); sha = hashlib.sha256(b).hexdigest(); p = os.path.join(self.root, sha + ".bin")
        if os.path.exists(p):
            if os.path.getsize(p) != len(b) or open(p, "rb").read() != b:
                raise RuntimeError(f"content-address collision/corruption at {p}")
            self.reused += 1
        else:
            tmp = p + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(b)
            os.replace(tmp, p); self.written += 1; self.bytes_written += len(b)
        return {"sha256": sha, "bytes": len(b), "dtype": str(t.dtype), "shape": list(t.shape), "path": os.path.relpath(p, os.path.dirname(self.root))}


class Inventory:
    """Streaming per-process raw inventory (JSONL, append-only); the terminal result binds its sha and count."""

    def __init__(self, path):
        self.path = path; self.n = 0
        open(self.path, "x").close()

    def seal(self, path, obj, kind, case_id, extra=None):
        body = dict(obj); body.pop("record_sha256", None)
        body["record_sha256"] = canonical_sha(body)
        if os.path.exists(path):
            raise FileExistsError(path)
        with open(path, "x") as f:
            json.dump(body, f, allow_nan=True)
        entry = {"kind": kind, "case_id": case_id, "path": os.path.relpath(path, os.path.dirname(self.path)), "bytes": os.path.getsize(path),
                 "file_sha256": sha256_file(path), "record_sha256": body["record_sha256"], "utc": utc()}
        if extra:
            entry.update(extra)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        self.n += 1
        return body["record_sha256"]


def per_head(d):
    f = d.flatten(1); return torch.sqrt((f * f).mean(dim=1)).tolist(), f.abs().max(dim=1).values.tolist()


def ref_record(surface, depth, N_, R, case_id, stratum, native_ref, c2_sha, native_repeat_shas, fixture_id, instance, kind_key):
    """Candidate-blind reference record (no C0 fields by construction)."""
    N64, R64 = N_.detach().cpu().to(torch.float64), R.detach().cpu().to(torch.float64)
    rms1, mx1 = per_head(N64 - R64); rms2, mx2 = per_head(R64)
    fin = lambda t: bool(torch.isfinite(t).all())  # noqa: E731
    return {"case_id": case_id, "fixture_id": fixture_id, "instance": instance, "kind_key": kind_key, "stratum": stratum, "surface": surface, "depth": int(depth),
            "reference_valid": bool(fin(N64) and fin(R64)), "rms_C1_C2": rms1, "maxabs_C1_C2": mx1, "rms_C2": rms2, "maxabs_C2": mx2,
            "native_sha256": native_ref["sha256"], "native_tensor": native_ref, "native_dtype": str(N_.dtype), "c2_sha256": c2_sha,
            "native_repeat_sha256s": list(native_repeat_shas), "c2_regenerable_on_cpu": True}


def metrics_record(surface, depth, Y, N_, R, case_id, stratum, cand_ref, native_ref, c2_sha, execution):
    """Final per-observation record: candidate metrics + the SAME reference fields/hashes as the candidate-blind record."""
    Y64, N64, R64 = Y.detach().cpu().to(torch.float64), N_.detach().cpu().to(torch.float64), R.detach().cpu().to(torch.float64)
    rms0, mx0 = per_head(Y64 - R64); rms1, mx1 = per_head(N64 - R64); rms2, mx2 = per_head(R64)
    fin = lambda t: bool(torch.isfinite(t).all())  # noqa: E731
    return {"case_id": case_id, "stratum": stratum, "surface": surface, "depth": int(depth), "metrics_valid": bool(fin(Y64)), "reference_valid": bool(fin(N64) and fin(R64)),
            "rms_C0_C2": rms0, "maxabs_C0_C2": mx0, "rms_C1_C2": rms1, "maxabs_C1_C2": mx1, "rms_C2": rms2, "maxabs_C2": mx2,
            "candidate_sha256": cand_ref["sha256"], "native_sha256": native_ref["sha256"], "c2_sha256": c2_sha,
            "candidate_tensor": cand_ref, "native_tensor": native_ref, "candidate_dtype": str(Y.dtype), "native_dtype": str(N_.dtype), "heads": int(Y64.shape[0]),
            "execution": execution}


class Backend:
    """Real GPU backend: owns the persistent component storage laid out like the deployed route and runs the public boot sequence."""

    def __init__(self, device, layers):
        import lumo_flywheel_serving.fr10_gdn_tree_kernel as kernel  # noqa: F401  (import only after identity refusal passed)
        from lumo_flywheel_serving.fr13_tree_conv_fused import build_tree_conv_state_src_indices
        from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as native_mod
        if layers != 48:
            raise RuntimeError("the deployed component lifecycle binds exactly 48 GDN layers (16 alias classes x 3 SSI groups)")
        self.kernel, self.native_mod, self.device, self.L = kernel, native_mod, device, layers
        dt = torch.bfloat16
        self.pages = build_alias_pages(device, ROWS)                       # 16 raws + 48 conv views + 48 ssm views (identity-preserving tuples)
        self.raws, self.conv_banks, self.banks = self.pages["raws"], self.pages["conv_banks"], self.pages["ssm_banks"]
        self.ring_k = torch.zeros((layers, CAPACITY, N, H, K), dtype=dt, device=device); self.ring_v = torch.zeros((layers, CAPACITY, N, HV, V), dtype=dt, device=device)
        self.ring_a = torch.zeros((layers, CAPACITY, N, HV), dtype=dt, device=device); self.ring_b = torch.zeros((layers, CAPACITY, N, HV), dtype=dt, device=device)
        self.spec_idx = torch.zeros((layers, CAPACITY, 32), dtype=torch.int32, device=device)           # commit SSI (48,B,32): col 0 running row, cols 1..31 scratch row
        for l in range(layers):
            self.spec_idx[l, :, 0] = run_row(l); self.spec_idx[l, :, 1:] = scratch_row(l)
        self.prev_lens = torch.zeros((CAPACITY,), dtype=torch.int32, device=device)
        self.A_logs = torch.zeros((layers, HV), dtype=torch.float32, device=device); self.dt_biases = torch.zeros((layers, HV), dtype=torch.float32, device=device)
        self.accepted_paths = torch.zeros((CAPACITY, 16), dtype=torch.int32, device=device); self.accepted_lens = torch.zeros((CAPACITY,), dtype=torch.int32, device=device)
        self.flags = torch.zeros((2,), dtype=torch.int32, device=device)
        self.strict, self.visible = masks_from_parent(kernel._FR13_FIXED32_PARENT, device)
        self.out = torch.zeros((layers, N, HV, V), dtype=dt, device=device)
        self.g0 = torch.zeros((N, HV), dtype=torch.float32, device=device); self.beta0 = torch.zeros((N, HV), dtype=torch.float32, device=device)
        self.c1_bank = torch.zeros((2, HV, V, K), dtype=torch.float32, device=device)   # independent scratch for the native reference
        self.scale = float(K ** -0.5)
        self.layer_order = tuple(layer_name(l) for l in range(layers))
        # builder-owned SSI groups (three, one per 16-layer group); registered ONCE by boot_lifecycle before any preseed
        self.ssi_groups = tuple(torch.zeros((CAPACITY, 32), dtype=torch.int32, device=device) for _ in range(SSI_GROUPS))
        # conv commit sources: 48 distinct bf16 stagings (B*36, 10240) + static state-src gather table (int64, 32*34)
        self.source_stagings = tuple(torch.zeros((CAPACITY * SOURCE_ROWS, CONV_C), dtype=dt, device=device) for _ in range(layers))
        self.state_src = build_tree_conv_state_src_indices(parent=list(kernel._FR13_FIXED32_PARENT), width=4, state_len=CONV_L, device=device).contiguous()
        self.boot, self.preseed, self.conv_contract, self.conv_audit, self.warm, self.warm_counters = boot_lifecycle(kernel, self)
        self.identity = self.pointer_identity()

    def pointer_identity(self):
        tens = {"banks": self.banks, "conv_banks": self.conv_banks, "raws": self.raws, "ssi_groups": self.ssi_groups, "source_stagings": self.source_stagings, "state_src": (self.state_src,),
                "ring_k": (self.ring_k,), "ring_v": (self.ring_v,), "ring_a": (self.ring_a,), "ring_b": (self.ring_b,), "spec_idx": (self.spec_idx,),
                "A_logs": (self.A_logs,), "dt_biases": (self.dt_biases,), "accepted_paths": (self.accepted_paths,), "accepted_lens": (self.accepted_lens,), "flags": (self.flags,), "out": (self.out,)}
        return {k: [(int(t.data_ptr()), tuple(t.stride()), tuple(t.shape), str(t.dtype), int(t.storage_offset()), id(t)) for t in v] for k, v in tens.items()}

    def load_instances(self, instances):
        for l, inp in enumerate(instances):
            self.A_logs[l].copy_(inp["A_log"].to(self.device)); self.dt_biases[l].copy_(inp["dt_bias"].to(self.device))

    def restore_O0(self, instances, control_seed):
        """In-place content restore on the page-shared banks: each layer touches ONLY its own alias-rank rows (running, scratch, controls) and row 0."""
        for l, inp in enumerate(instances):
            b = self.banks[l]; b[0].zero_(); b[run_row(l)].copy_(inp["S0"].to(self.device)); b[scratch_row(l)].zero_()
            for r in control_rows(l):   # distinct finite sentinel patterns per untouched control row (layer-specific value, rank-specific row)
                b[r].fill_(float(1000 * (l + 1) + r + control_seed))
        self.ring_k.zero_(); self.ring_v.zero_(); self.ring_a.zero_(); self.ring_b.zero_(); self.flags.zero_(); self.out.zero_()

    def scan(self, instances):
        kernel = self.kernel
        for l, inp in enumerate(instances):
            q, k, v, a, b = (inp[n].to(self.device).contiguous() for n in ("q", "k", "v", "a", "b"))
            kernel.launch_tree_gdn_prepared(q=q, k=k, v=v, g=self.g0, beta=self.beta0, raw_a=a, raw_b=b, A_log=self.A_logs[l], dt_bias=self.dt_biases[l],
                                            h0=self.banks[l], h0_indices=self.spec_idx[l], h0_num_accepted_tokens=self.prev_lens, h0_is_bank=True, h0_index_row=0, h0_batch_index=0,
                                            h0_use_accepted_column=False, n_actual=N, n_pad=N, strict_mask=self.strict, visible_mask=self.visible, out=self.out[l], state=None,
                                            output_scale=self.scale, use_qk_l2norm_in_kernel=True, ring_k=self.ring_k[l, 0], ring_v=self.ring_v[l, 0], ring_a=self.ring_a[l, 0],
                                            ring_b=self.ring_b[l, 0], staging_flags=self.flags, staging_rows=1)
        torch.cuda.synchronize()

    def route_stamp(self):
        """Actual executed scan route stamp(s) from the kernel's subtree cache (receipt, not a claim)."""
        out = []
        for key, st in getattr(self.kernel, "_FR13_SUBTREE_CACHE", {}).items():
            le = st.get("last_executed_gdn") if isinstance(st, dict) else None
            out.append({"cache_key": str(key)[:200], "schedule": str(st.get("schedule"))[:400] if isinstance(st, dict) else None, "critical": st.get("critical") if isinstance(st, dict) else None,
                        "last_executed_gdn": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in le.items()} if isinstance(le, dict) else le})
        return out

    def ring_bytes_equal(self, instances):
        return all(torch.equal(self.ring_k[l, 0], instances[l]["k"].to(self.device)) and torch.equal(self.ring_v[l, 0], instances[l]["v"].to(self.device))
                   and torch.equal(self.ring_a[l, 0], instances[l]["a"].to(self.device)) and torch.equal(self.ring_b[l, 0], instances[l]["b"].to(self.device)) for l in range(self.L))

    def publish(self, accepted_nodes):
        drafts = accepted_nodes[1:]
        self.accepted_paths.zero_(); self.accepted_lens.zero_()
        if drafts:
            self.accepted_paths[0, :len(drafts)].copy_(torch.tensor(drafts, dtype=torch.int32, device=self.device))
        self.accepted_lens[0] = len(drafts)
        before = dict(self.kernel.fixed32_committer_counters())
        # NOTE: bank_off16 is a LEGACY argument of the generic entry; in fixed32 mode the replay dispatches to the preseeded
        # persistent committer bound to `banks_list` and does not read bank_off16 as a pointer table.
        self.kernel.launch_tree_gdn_replay_all_layers(
            bank_anchor=self.banks[0], bank_off16=self.spec_idx, bank_shape=tuple(self.banks[0].shape), bank_stride=int(self.banks[0].stride(0)), spec_state_indices=self.spec_idx,
            prev_lens=self.prev_lens, accepted_paths=self.accepted_paths, accepted_lens=self.accepted_lens, k_rings=self.ring_k, k_norm_rings=None, gate_rings=None,
            v_rings=self.ring_v, a_rings=self.ring_a, b_rings=self.ring_b, A_logs=self.A_logs, dt_biases=self.dt_biases, num_layers=self.L, num_spec_decodes=1,
            output_scale=self.scale, use_qk_l2norm_in_kernel=True, runrow_commit=True, runrow_init=True, burn_node_bank=False, banks_list=self.banks)
        torch.cuda.synchronize(); after = dict(self.kernel.fixed32_committer_counters())
        return int(after["actual_replays_enqueued"]) - int(before["actual_replays_enqueued"])

    def counters(self):
        return dict(self.kernel.fixed32_committer_counters())

    def replay_stale(self):
        """Replay with whatever accepted path/len tensors currently hold (stale-metadata negative)."""
        before = self.counters(); self.kernel.launch_tree_gdn_replay_all_layers(
            bank_anchor=self.banks[0], bank_off16=self.spec_idx, bank_shape=tuple(self.banks[0].shape), bank_stride=int(self.banks[0].stride(0)), spec_state_indices=self.spec_idx,
            prev_lens=self.prev_lens, accepted_paths=self.accepted_paths, accepted_lens=self.accepted_lens, k_rings=self.ring_k, k_norm_rings=None, gate_rings=None,
            v_rings=self.ring_v, a_rings=self.ring_a, b_rings=self.ring_b, A_logs=self.A_logs, dt_biases=self.dt_biases, num_layers=self.L, num_spec_decodes=1,
            output_scale=self.scale, use_qk_l2norm_in_kernel=True, runrow_commit=True, runrow_init=True, burn_node_bank=False, banks_list=self.banks)
        torch.cuda.synchronize(); return int(self.counters()["actual_replays_enqueued"]) - int(before["actual_replays_enqueued"])

    def swap_rings(self, l1, l2):
        tmp = self.ring_k[l1, 0].clone(); self.ring_k[l1, 0].copy_(self.ring_k[l2, 0]); self.ring_k[l2, 0].copy_(tmp)

    def poison_offpath_rings(self, off_nodes):
        for l in range(self.L):
            self.ring_k[l, 0, off_nodes] = 12345.0; self.ring_v[l, 0, off_nodes] = -12345.0; self.ring_a[l, 0, off_nodes] = 777.0; self.ring_b[l, 0, off_nodes] = -777.0

    def published_state(self, l):
        return self.banks[l][run_row(l)].clone().cpu()

    def control_rows_intact(self, l, control_seed):
        b = self.banks[l]
        return all(torch.equal(b[r], torch.full_like(b[r], float(1000 * (l + 1) + r + control_seed))) for r in control_rows(l)) and bool(torch.equal(b[0], torch.zeros_like(b[0])))

    def staging_neutral_tail(self, accepted_len):
        st = self.kernel._FR13_FIXED32_COMMITTER_FAST_ROUTE["state"]["states_by_batch"][1]
        tail = slice(accepted_len + 1, 16)   # position 0 = root, 1..len = drafts, > len neutral
        return bool(torch.equal(st["abuf"][:, tail], torch.full_like(st["abuf"][:, tail], -1e4)) and torch.equal(st["bbuf"][:, tail], torch.zeros_like(st["bbuf"][:, tail]))
                    and torch.equal(st["kbuf"][:, tail], torch.zeros_like(st["kbuf"][:, tail])) and torch.equal(st["vbuf"][:, tail], torch.zeros_like(st["vbuf"][:, tail])))

    def native_chain(self, inp, nodes):
        """C1: stock native op token by token along root-inclusive `nodes` on an independent scratch bank; returns (state, last_out)."""
        fn = self.native_mod.fused_sigmoid_gating_delta_rule_update
        self.c1_bank.zero_(); self.c1_bank[1].copy_(inp["S0"].to(self.device)); last = None
        for node in nodes:
            q, k, v, a, b = (inp[n][node:node + 1].to(self.device).unsqueeze(0) for n in ("q", "k", "v", "a", "b"))
            cu = torch.tensor([0, 1], dtype=torch.int32, device=self.device); ssi = torch.ones((1, 1), dtype=torch.int32, device=self.device)
            out, _ = fn(A_log=inp["A_log"].to(self.device), a=a, b=b, dt_bias=inp["dt_bias"].to(self.device), q=q, k=k, v=v, scale=self.scale, initial_state=self.c1_bank,
                        inplace_final_state=True, cu_seqlens=cu, ssm_state_indices=ssi, num_accepted_tokens=None, use_qk_l2norm_in_kernel=True)
            last = out.reshape(HV, V)
        torch.cuda.synchronize()
        return self.c1_bank[1].clone().cpu(), (last.clone().cpu() if last is not None else None)

    def scan_output(self, l, node):
        return self.out[l, node].clone().cpu()


def _case_path(raw_dir, cid):
    return os.path.join(raw_dir, cid.split("|", 1)[1].replace("|", "__") + ".json")


INIT_GATE_NAME, CAL_GATE_NAME = "Q1.2b-INIT", "Q1.2b"
HELD_OUT_BLOCK = "evaluation"                       # the ONLY non-calibration block: manifest block value; policy domain key "evaluation_held_out"
SUPPORTED_BLOCKS = ("calibration", HELD_OUT_BLOCK)


def block_authorization(gate_doc, pol, manifest, args):
    """v2.2: block-level authorization + complete-coverage facts (in ADDITION to authorize_scope, never instead of it).

    * calibration: gate reviewed_scope.block must be "calibration" (v2.1 semantics; the launcher checked it, the runner now does too).
    * evaluation (held out): reviewed_scope.block == "evaluation"; reviewed_scope.negatives == 0 AND --negatives 0 (negative
      mutations are calibration-only); --limit 0 (every held-out fixture runs); reviewed_scope.held_out_fixture_ids == the
      manifest's evaluation fixture ids == the policy domain's evaluation_held_out ids (non-empty, unique, typed);
      reviewed_scope.calibration_power_run_id names the accepted calibration run whose negative power backs this block;
      reviewed_hashes.runner binds THIS runner's bytes.
    * any other block string is refused.  Returns (problems, facts); the caller refuses on any problem before Backend construction."""
    sc = gate_doc.get("reviewed_scope") or {}; rh = gate_doc.get("reviewed_hashes") or {}
    entries = [e for e in (manifest.get("fixtures") or []) if isinstance(e, dict) and e.get("block") == args.block]
    ids = [e.get("fixture_id") for e in entries]
    dom = ((pol.get("domain") or {}).get("q1_2b") or {}).get("fixture_ids") or {}
    facts = {"block": args.block, "supported": args.block in SUPPORTED_BLOCKS, "manifest_fixture_ids": ids, "gate_scope_block": sc.get("block"),
             "gate_scope_negatives": sc.get("negatives"), "args_negatives": int(args.negatives), "args_limit": int(args.limit)}
    pr = []
    if args.block not in SUPPORTED_BLOCKS:
        pr.append(f"unsupported block {args.block!r}: supported blocks are {SUPPORTED_BLOCKS}"); facts["complete_coverage"] = False; return pr, facts
    if sc.get("block") != args.block:
        pr.append(f"gate reviewed_scope.block {sc.get('block')!r} does not authorize block {args.block!r}")
    if not entries or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i.startswith(args.block + "/") or not isinstance(e.get("sha256"), str) or len(e["sha256"]) != 64
                                                          for i, e in zip(ids, entries)):
        pr.append(f"manifest has no complete typed unique fixture set for block {args.block!r}: {ids}")
    if args.block == HELD_OUT_BLOCK:
        dom_ids = dom.get("evaluation_held_out"); named = sc.get("held_out_fixture_ids")
        facts.update({"policy_domain_held_out_ids": dom_ids, "gate_named_held_out_ids": named, "calibration_power_run_id": sc.get("calibration_power_run_id")})
        if not isinstance(dom_ids, list) or not dom_ids or sorted(dom_ids) != sorted(ids):
            pr.append(f"policy domain evaluation_held_out ids {dom_ids} != manifest held-out ids {sorted(ids)}")
        if not isinstance(named, list) or not named or len(set(named)) != len(named) or sorted(named) != sorted(ids):
            pr.append(f"gate reviewed_scope.held_out_fixture_ids {named} does not name exactly the manifest held-out ids {sorted(ids)}")
        if str(sc.get("negatives")) != "0" or int(args.negatives) != 0:
            pr.append("negative mutations are calibration-only: held-out requires --negatives 0 and reviewed_scope.negatives 0")
        if int(args.limit) != 0:
            pr.append("held-out requires complete coverage: --limit must be 0")
        if not isinstance(sc.get("calibration_power_run_id"), str) or not sc.get("calibration_power_run_id"):
            pr.append("gate reviewed_scope.calibration_power_run_id must name the accepted calibration run (negative power source)")
        if rh.get("runner") != sha256_file(os.path.abspath(__file__)):
            pr.append("held-out gate does not bind this runner's bytes")
    facts["complete_coverage"] = not pr and int(args.limit) == 0
    return pr, facts


def authorize_scope(gate_doc, pol, args):
    """Scope-specific authorization from the launch-time gate snapshot.

    * calibration (default): the gate must be GATE-Q1.2b and the policy must be EXTERNALLY authorized for numerical use
      (evaluator v2.1 `_approved_ok`: gate.approved + reviewed_hashes.policy == loaded sha + gate == "Q1.2b").
    * --init-only: the gate must be GATE-Q1.2b-INIT with approved:true, reviewed_scope.init_only:true, the policy is used for
      INTEGRITY and DOMAIN checks only (`_integrity_ok`, policy sha == the gate-bound expected sha); numerical authorization is
      neither required nor claimed (the evaluator is not modified or relaxed; no case is evaluated in this scope).
    Returns {"scope", "ok", "problems", ...}; the caller refuses on any problem before Backend construction."""
    rh = gate_doc.get("reviewed_hashes") or {}; sc = gate_doc.get("reviewed_scope") or {}
    out = {"scope": "init_only" if args.init_only else "calibration", "gate": gate_doc.get("gate"), "approved": gate_doc.get("approved") is True, "approved_run_id": gate_doc.get("approved_run_id"),
           "policy_integrity_ok": bool(pol.get("_integrity_ok")), "policy_numerically_authorized": bool(pol.get("_approved_ok")), "problems": []}
    if not pol.get("_integrity_ok"):
        out["problems"].append(f"policy v2.1 integrity failed: {pol.get('_load_problems')}")
    if rh.get("policy") != args.expected_policy_sha256 or pol.get("_loaded_sha256") != args.expected_policy_sha256:
        out["problems"].append("policy sha not bound by the gate snapshot / launcher")
    if gate_doc.get("approved") is not True or not gate_doc.get("approved_run_id"):
        out["problems"].append("gate not approved / no approved_run_id")
    if args.init_only:
        if gate_doc.get("gate") != INIT_GATE_NAME:
            out["problems"].append(f"init-only scope requires gate {INIT_GATE_NAME!r}, got {gate_doc.get('gate')!r}")
        if sc.get("init_only") is not True:
            out["problems"].append("init gate reviewed_scope.init_only must be true")
        if rh.get("runner") != sha256_file(os.path.abspath(__file__)):
            out["problems"].append("init gate does not bind this runner's bytes")
        out["numerical_authorization_claimed"] = False
    else:
        if gate_doc.get("gate") != CAL_GATE_NAME:
            out["problems"].append(f"calibration scope requires gate {CAL_GATE_NAME!r}, got {gate_doc.get('gate')!r} (an init approval is NOT a calibration approval)")
        if sc.get("init_only") is True:
            out["problems"].append("calibration scope refuses an init-only gate")
        if not pol.get("_approved_ok"):
            out["problems"].append(f"policy v2.1 not numerically authorized for calibration: {pol.get('_auth_problems')}")
    out["ok"] = not out["problems"]
    return out


def gate_approval(gate_snapshot_path):
    """External authorization record derived from the parent's gate snapshot (policy bytes stay immutable)."""
    g = json.load(open(gate_snapshot_path))
    rh = g.get("reviewed_hashes") or {}
    return {"approved": g.get("approved") is True, "policy_sha256": rh.get("policy"), "stage": g.get("gate"), "expected_observations_sha256": rh.get("expected_observations"),
            "approved_run_id": g.get("approved_run_id"), "approved_utc": g.get("approved_utc"), "source": os.path.basename(gate_snapshot_path)}


def run_fixture(be, fixture, entry, cases_by_id, args, out_dir, store, inv, policy):
    """Reference-first execution of one fixture. Returns the fixture record (structural flags + case/negative indexes)."""
    meta, instances, refs = fixture["meta"], fixture["instances"], fixture["refs"]
    fid = meta["fixture_id"]; L = len(instances); paths = meta["accepted_paths"]; by_pid = {p["path_id"]: p for p in paths}
    rec = {"fixture_id": fid, "block": meta["block"], "stratum": meta["stratum"], "layers": L, "structural": {}, "phase": {}, "cases": [], "ref_cases": [], "negatives": [],
           "repeat_hashes": [], "native_repeats": [], "eligibility": {"eligible": 0, "uncovered": 0, "malformed": 0}}
    raw_dir = os.path.join(out_dir, "raw", fid.replace("/", "__")); ref_dir = os.path.join(out_dir, "ref", fid.replace("/", "__")); elig_dir = os.path.join(out_dir, "eligibility", fid.replace("/", "__"))
    for d in (raw_dir, ref_dir, elig_dir):
        os.makedirs(d, exist_ok=True)
    exec_base = {"process_tag": args.process_tag, "fixture_id": fid, "fixture_sha256": entry["sha256"], "runner_sha256": sha256_file(os.path.abspath(__file__))}
    # ---------------- PHASE R: references first (candidate-blind) ----------------
    rec["phase"]["R_started_utc"] = utc()
    c2 = []
    for l, inp in enumerate(instances):
        st, ou = FX.c2_node_refs(inp, float(meta["scale"]))
        ok = all(hashlib.sha256(st[n].numpy().tobytes()).hexdigest() == refs[l]["node_state_sha256"][str(n)] and hashlib.sha256(ou[n].numpy().tobytes()).hexdigest() == refs[l]["node_out_sha256"][str(n)] for n in range(N))
        if not ok:
            rec["structural"]["c2_hash_matches_fixture"] = False; rec["status"] = "C2_REFERENCE_HASH_MISMATCH"; return rec
        c2.append((st, ou))
    rec["structural"]["c2_hash_matches_fixture"] = True
    be.load_instances(instances)
    c1_state, c1_out, c1_state_ref, c1_out_ref = [], [], [], []   # repeat 0 tensors + store refs
    census = []   # per repeat: {"repeat": r, "state_sha256": {l: {node: sha}}, "out_sha256": {...}}
    for rep in range(args.repeats):
        cs_sha, co_sha = [], []
        for l in range(L):
            s_l, o_l = {}, {}
            for node in range(N):
                s_, o_ = be.native_chain(instances[l], FX.ancestors(node))
                s_l[str(node)] = sha_t(s_); o_l[str(node)] = sha_t(o_)
                if rep == 0:
                    if l == len(c1_state):
                        c1_state.append({}); c1_out.append({}); c1_state_ref.append({}); c1_out_ref.append({})
                    c1_state[l][node] = s_; c1_out[l][node] = o_
                    c1_state_ref[l][node] = store.put(s_); c1_out_ref[l][node] = store.put(o_)
                else:
                    store.put(s_); store.put(o_)   # dedup if bitwise equal; a differing repeat leaves its own content object
            cs_sha.append(s_l); co_sha.append(o_l)
        census.append({"repeat": rep, "state_sha256": cs_sha, "out_sha256": co_sha})
    rec["native_repeats"] = census
    rec["structural"]["c1_within_process_bitwise"] = all(c["state_sha256"] == census[0]["state_sha256"] and c["out_sha256"] == census[0]["out_sha256"] for c in census)
    rep_shas = lambda l, node, kind: [c[kind][l][str(node)] for c in census]  # noqa: E731
    # candidate-blind reference records + reference-only eligibility for EVERY expected case of this fixture
    for l in range(L):
        for node in range(N):
            cid = f"{fid}|L{l}|out|node{node:02d}"; case = cases_by_id[cid]
            r = ref_record("output", case["depths"][0], c1_out[l][node], c2[l][1][node], cid, meta["stratum"], c1_out_ref[l][node], case["reference_sha256"], rep_shas(l, node, "out_sha256"), fid, l, f"out|node{node:02d}")
            if hashlib.sha256(c2[l][1][node].numpy().tobytes()).hexdigest() != case["reference_sha256"]:
                r["reference_valid"] = False; r["c2_expected_mismatch"] = True
            rsha = inv.seal(_case_path(ref_dir, cid), r, "ref", cid, {"fixture_id": fid})
            el = PE21.reference_only_eligibility(policy, case, [r], bound_cases=cases_by_id, fixture_sha256=entry["sha256"]); el["ref_record_sha256"] = rsha
            inv.seal(_case_path(elig_dir, cid), el, "eligibility", cid, {"fixture_id": fid})
            rec["ref_cases"].append({"case_id": cid, "eligible": bool(el.get("eligible")), "status": el.get("status"), "uncovered_all": bool(el.get("uncovered_all"))})
        for p in paths:
            cid = f"{fid}|L{l}|state|{p['path_id']}"; case = cases_by_id[cid]; leaf = p["leaf_node"]
            r = ref_record("state", case["depths"][0], c1_state[l][leaf], c2[l][0][leaf], cid, meta["stratum"], c1_state_ref[l][leaf], case["reference_sha256"], rep_shas(l, leaf, "state_sha256"), fid, l, f"state|{p['path_id']}")
            if hashlib.sha256(c2[l][0][leaf].numpy().tobytes()).hexdigest() != case["reference_sha256"]:
                r["reference_valid"] = False; r["c2_expected_mismatch"] = True
            rsha = inv.seal(_case_path(ref_dir, cid), r, "ref", cid, {"fixture_id": fid})
            el = PE21.reference_only_eligibility(policy, case, [r], bound_cases=cases_by_id, fixture_sha256=entry["sha256"]); el["ref_record_sha256"] = rsha
            inv.seal(_case_path(elig_dir, cid), el, "eligibility", cid, {"fixture_id": fid})
            rec["ref_cases"].append({"case_id": cid, "eligible": bool(el.get("eligible")), "status": el.get("status"), "uncovered_all": bool(el.get("uncovered_all"))})
    for x in rec["ref_cases"]:
        rec["eligibility"]["eligible" if x["eligible"] else ("malformed" if x["status"] == "MALFORMED_EVIDENCE" else "uncovered")] += 1
    rec["phase"]["R_sealed_utc"] = utc()
    # ---------------- PHASE C: candidate (only now is the candidate route launched) ----------------
    clean_pub_sha = {}   # path_id -> [sha per layer] from repeat 0 (for N5 causal invariance)
    for rep in range(args.repeats):
        seed = 17 + rep
        be.restore_O0(instances, seed); t_scan = utc(); be.scan(instances)
        if "C_first_scan_utc" not in rec["phase"]:
            rec["phase"]["C_first_scan_utc"] = t_scan; rec["route_stamp_after_first_scan"] = be.route_stamp() if hasattr(be, "route_stamp") else None
        rings_ok = be.ring_bytes_equal(instances)
        out_h = [sha_t(be.out[l]) for l in range(L)]
        rep_rec = {"repeat": rep, "ring_bytes_exact": bool(rings_ok), "scan_out_sha256": out_h, "scan_node_sha256": [], "publish": {}}
        for l in range(L):
            node_shas = {}
            for node in range(N):
                Y = be.scan_output(l, node); cref = store.put(Y); node_shas[str(node)] = cref["sha256"]
                if rep == 0:
                    cid = f"{fid}|L{l}|out|node{node:02d}"; case = cases_by_id[cid]
                    execution = {**exec_base, "repeat": rep, "utc": utc(), "instance": l, "node": node, "expected_reference_sha256": case["reference_sha256"]}
                    rmet = metrics_record("output", case["depths"][0], Y, c1_out[l][node], c2[l][1][node], cid, meta["stratum"], cref, c1_out_ref[l][node], case["reference_sha256"], execution)
                    msha = inv.seal(_case_path(raw_dir, cid), rmet, "metrics", cid, {"fixture_id": fid, "repeat": rep})
                    rec["cases"].append({"case_id": cid, "kind": "output", "record_sha256": msha, "candidate_sha256": cref["sha256"], "native_sha256": c1_out_ref[l][node]["sha256"], "nonfinite_candidate": nonfinite(Y)})
            rep_rec["scan_node_sha256"].append(node_shas)
        for p in paths:
            nodes = p["nodes"]; be.restore_O0(instances, seed); be.scan(instances)   # identical scan (deterministic); rings regenerated
            replays = be.publish(nodes)
            pub = [be.published_state(l) for l in range(L)]
            prefs = [store.put(x) for x in pub]
            ident_ok = be.pointer_identity() == be.identity
            ctrl_ok = all(be.control_rows_intact(l, seed) for l in range(L))
            tail_ok = be.staging_neutral_tail(p["accepted_len_excl_root"])
            c1_eq = all(torch.equal(pub[l], c1_state[l][p["leaf_node"]]) for l in range(L))
            rep_rec["publish"][p["path_id"]] = {"replays_enqueued": int(replays), "pointer_identity_unchanged": bool(ident_ok), "control_rows_intact": bool(ctrl_ok),
                                                "staging_neutral_tail": bool(tail_ok), "published_equals_native_chain_bitwise": bool(c1_eq), "state_sha256": [x["sha256"] for x in prefs]}
            if rep == 0:
                clean_pub_sha[p["path_id"]] = [x["sha256"] for x in prefs]
                for l in range(L):
                    cid = f"{fid}|L{l}|state|{p['path_id']}"; case = cases_by_id[cid]
                    execution = {**exec_base, "repeat": rep, "utc": utc(), "instance": l, "path_id": p["path_id"], "accepted_nodes": nodes, "expected_reference_sha256": case["reference_sha256"], "replays_enqueued": int(replays)}
                    rmet = metrics_record("state", case["depths"][0], pub[l], c1_state[l][p["leaf_node"]], c2[l][0][p["leaf_node"]], cid, meta["stratum"], prefs[l], c1_state_ref[l][p["leaf_node"]], case["reference_sha256"], execution)
                    msha = inv.seal(_case_path(raw_dir, cid), rmet, "metrics", cid, {"fixture_id": fid, "repeat": rep})
                    rec["cases"].append({"case_id": cid, "kind": "state", "record_sha256": msha, "candidate_sha256": prefs[l]["sha256"], "native_sha256": c1_state_ref[l][p["leaf_node"]]["sha256"], "nonfinite_candidate": nonfinite(pub[l])})
        rec["repeat_hashes"].append(rep_rec)
        print(f"[q1.2b-v2 {args.process_tag}] {fid} rep{rep}: rings_exact={rings_ok} paths={len(paths)} replays_ok={all(v['replays_enqueued']==1 for v in rep_rec['publish'].values())} "
              f"ptr_ok={all(v['pointer_identity_unchanged'] for v in rep_rec['publish'].values())} ctrl_ok={all(v['control_rows_intact'] for v in rep_rec['publish'].values())} "
              f"tail_ok={all(v['staging_neutral_tail'] for v in rep_rec['publish'].values())} pub==C1={all(v['published_equals_native_chain_bitwise'] for v in rep_rec['publish'].values())}", flush=True)
    rec["phase"]["C_ended_utc"] = utc()
    r0 = rec["repeat_hashes"][0]
    rec["structural"]["within_process_bitwise"] = all(r["scan_out_sha256"] == r0["scan_out_sha256"] and r["scan_node_sha256"] == r0["scan_node_sha256"]
                                                     and all(r["publish"][k]["state_sha256"] == r0["publish"][k]["state_sha256"] for k in r0["publish"]) for r in rec["repeat_hashes"])
    rec["structural"]["ring_bytes_exact_all"] = all(r["ring_bytes_exact"] for r in rec["repeat_hashes"])
    rec["structural"]["replays_one_per_publish"] = all(v["replays_enqueued"] == 1 for r in rec["repeat_hashes"] for v in r["publish"].values())
    rec["structural"]["pointer_identity_unchanged"] = all(v["pointer_identity_unchanged"] for r in rec["repeat_hashes"] for v in r["publish"].values())
    rec["structural"]["control_rows_intact"] = all(v["control_rows_intact"] for r in rec["repeat_hashes"] for v in r["publish"].values())
    rec["structural"]["staging_neutral_tail"] = all(v["staging_neutral_tail"] for r in rec["repeat_hashes"] for v in r["publish"].values())
    rec["structural"]["no_nonfinite_candidate"] = all(c["nonfinite_candidate"] == 0 for c in rec["cases"])
    rec["structural"]["reference_phase_before_candidate"] = rec["phase"]["R_sealed_utc"] <= rec["phase"]["C_first_scan_utc"]
    rec["structural"]["expected_case_product_sealed"] = (len(rec["cases"]) == L * (N + len(paths)) == len(rec["ref_cases"]) and len({c["case_id"] for c in rec["cases"]}) == len(rec["cases"]))
    rec["findings"] = {"published_equals_native_chain_bitwise_all": all(v["published_equals_native_chain_bitwise"] for r in rec["repeat_hashes"] for v in r["publish"].values())}
    if args.negatives and meta["block"] == "calibration":
        rec["negatives"] = run_negatives(be, instances, meta, c2, c1_state, c1_state_ref, raw_dir, fid, args, store, inv, exec_base, clean_pub_sha)
        rec["structural"]["n5_poisoned_equals_clean_candidate"] = next(n for n in rec["negatives"] if n["tag"] == "N5_offpath_sentinels")["leaked_instances"] == 0
        rec["structural"]["negative_product_exact"] = (sorted(n["tag"] for n in rec["negatives"]) == sorted(["N1_sibling_substitution", "N3_ring_swap", "N4_stale_metadata", "N5_offpath_sentinels", "N7_incumbent_twice"])
                                                       and all(n.get("records_sealed") == n.get("records_expected") for n in rec["negatives"] if n["tag"] != "N7_incumbent_twice"))
    rec["structural_ok"] = all(rec["structural"].values())
    rec["status"] = "DONE"
    return rec


def run_negatives(be, instances, meta, c2, c1_state, c1_state_ref, raw_dir, fid, args, store, inv, exec_base, clean_pub_sha):
    L = len(instances); paths = meta["accepted_paths"]; by_id = {p["path_id"]: p for p in paths}
    if L < 2:
        raise RuntimeError("negative controls require >= 2 operator instances (N3 swaps instances 0 and 1)")
    target = by_id[N5_TARGET_PATH]; leaf = target["leaf_node"]
    neg = []
    def seal(tag, l, Y, extra):
        cref = store.put(Y)
        execution = {**exec_base, "negative": tag, "utc": utc(), "instance": l, "path_id": target["path_id"]}
        r = metrics_record("state", 1 + target["accepted_len_excl_root"], Y, c1_state[l][leaf], c2[l][0][leaf], f"NEG|{fid}|L{l}|{tag}", meta["stratum"], cref, c1_state_ref[l][leaf], hashlib.sha256(c2[l][0][leaf].numpy().tobytes()).hexdigest(), execution)
        r.update({"negative": tag, **extra})
        inv.seal(os.path.join(raw_dir, f"NEG__L{l}__{tag}.json"), r, "negative", r["case_id"], {"fixture_id": fid, "negative": tag})
        return cref["sha256"]
    # N1 sibling substitution: publish [0,1,4,9,15] but compare against the CORRECT [0,1,4,9,14] references
    be.restore_O0(instances, 99); be.scan(instances); be.publish(N1_SIBLING_PATH)
    n = 0
    for l in range(L):
        seal("N1_sibling_substitution", l, be.published_state(l), {"published_path": N1_SIBLING_PATH, "reference_path": target["nodes"]}); n += 1
    neg.append({"tag": "N1_sibling_substitution", "records_expected": L, "records_sealed": n, "expect": "paired bounds violated vs correct references"})
    # N3 ring swap: swap ring contents of two instances before publishing the correct path
    be.restore_O0(instances, 99); be.scan(instances); be.swap_rings(*N3_INSTANCES); be.publish(target["nodes"])
    n = 0
    for l in N3_INSTANCES:
        seal("N3_ring_swap", l, be.published_state(l), {"swapped_instances": list(N3_INSTANCES)}); n += 1
    neg.append({"tag": "N3_ring_swap", "records_expected": len(N3_INSTANCES), "records_sealed": n, "affected_instances": list(N3_INSTANCES), "expect": "instances 0/1 violate paired bounds"})
    # N4 stale metadata: previous publication's path left in place (no new copy) while intending n14
    be.restore_O0(instances, 99); be.scan(instances); be.publish(N4_PREVIOUS_PATH)
    be.restore_O0(instances, 99); be.scan(instances); be.replay_stale()
    n = 0
    for l in range(L):
        seal("N4_stale_metadata", l, be.published_state(l), {"stale_path": N4_PREVIOUS_PATH, "intended_path": target["nodes"]}); n += 1
    neg.append({"tag": "N4_stale_metadata", "records_expected": L, "records_sealed": n, "expect": "paired bounds violated (published n04 state vs intended n14 references)"})
    # N5 causal off-path invariance: poisoned publication must equal the SAME clean publication (repeat-0 C0 of n14) bitwise
    be.restore_O0(instances, 17); be.scan(instances)   # same control seed as repeat 0 -> same O0/rings as the clean publication
    off = [x for x in range(N) if x not in target["nodes"]]
    be.poison_offpath_rings(off); be.publish(target["nodes"])
    clean = clean_pub_sha.get(target["path_id"], [None] * L); leaked = 0; n = 0
    for l in range(L):
        Y = be.published_state(l); psha = sha_t(Y); eq = (psha == clean[l])
        leaked += 0 if eq else 1
        seal("N5_offpath_sentinels", l, Y, {"clean_candidate_sha256": clean[l], "poisoned_candidate_sha256": psha, "poisoned_equals_clean_candidate": bool(eq), "poisoned_nodes": off,
                                             "note": "causal invariant is poisoned C0 == clean C0; the paired C1/C2 fields here are diagnostic only"}); n += 1
    neg.append({"tag": "N5_offpath_sentinels", "records_expected": L, "records_sealed": n, "leaked_instances": int(leaked), "expect": "no leak: poisoned C0 == clean C0 bitwise on every instance"})
    # N7 incumbent twice: no replay at all -> counters unchanged -> the harness must reject
    be.restore_O0(instances, 99); be.scan(instances)
    before = be.counters(); after = be.counters(); d = int(after["actual_replays_enqueued"]) - int(before["actual_replays_enqueued"])
    neg.append({"tag": "N7_incumbent_twice", "replays_enqueued": d, "expect": "0 replays -> REJECT (engagement check)", "harness_rejects": bool(d == 0)})
    return neg


def helper_hashes():
    out = {}
    for f in HELPER_FILES:
        p = os.path.join(HERE, f)
        out[f] = sha256_file(p) if os.path.exists(p) else None
    for rel in REPO_HELPER_FILES:
        out[rel] = sha256_file(os.path.join(ROOT, rel))
    return out


def projected_archive_bytes(layers, fixtures, repeats):
    state = HV * V * K * 4; out = HV * V * 2
    native = layers * N * (state + out)                  # C1 per fixture (dedup across repeats if bitwise equal)
    cand = layers * (N * out + 28 * state)               # scan outputs + published states per fixture per repeat (dedup if bitwise equal)
    neg = layers * state * 3 + 2 * state                 # N1, N4, N5 per layer + N3 two instances
    return {"per_fixture_native_bytes": native, "per_fixture_candidate_bytes_per_repeat": cand, "per_fixture_negative_bytes": neg,
            "worst_case_total_bytes_two_processes": 2 * fixtures * (native * repeats + cand * repeats + neg), "dedup_expected_total_bytes": fixtures * (native + cand + neg)}


def attest(args, kernel_path, native_mod, kernel):
    import inspect
    fn = native_mod.fused_sigmoid_gating_delta_rule_update
    return {"schema": "lumo.review-response.q1-2b-attestation.v2", "utc": utc(), "process_tag": args.process_tag, "pid": os.getpid(),
            "pid_note": "PID is evidence only: fresh --entrypoint containers all report pid 1; process independence is bound to container identity",
            "hostname_in_container": socket.gethostname(), "container_name_env": os.environ.get("Q12B_CONTAINER_NAME"),
            "container_cgroup_sha256": hashlib.sha256(open("/proc/self/cgroup", "rb").read()).hexdigest() if os.path.exists("/proc/self/cgroup") else None,
            "process_start_monotonic_ns": time.monotonic_ns(), "image_id_arg": args.image_id, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "triton": __import__("triton").__version__,
            "vllm": getattr(__import__("vllm"), "__version__", None), "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "kernel_module_sha256": sha256_file(kernel_path), "topology_sha256": sha256_file(os.path.join(ROOT, "scripts", "fr13_fixed32_topology.py")),
            "native_module_sha256": sha256_file(native_mod.__file__), "native_function_source_sha256": hashlib.sha256(inspect.getsource(fn).encode()).hexdigest(),
            "native_dispatch": {"module": native_mod.__name__, "file": native_mod.__file__, "function": fn.__qualname__, "triton_kernels": sorted(n for n in dir(native_mod) if n.startswith("fused_sigmoid_gating") and n != fn.__name__)},
            "env_subset": {k: os.environ.get(k) for k in list(REQUIRED_ENV) + ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONPATH"]},
            "runner_sha256": sha256_file(os.path.abspath(__file__)), "oracle_sha256": sha256_file(os.path.join(HERE, "q1_oracle.py")), "fixtures_tool_sha256": sha256_file(os.path.join(HERE, "q1_2b_fixtures.py")),
            "helper_hashes": helper_hashes(),
            "effective_topology": {"tree_profile": str(getattr(kernel, "_FR13_FIXED32_TREE_PROFILE", None)), "mode": kernel._FR13_FIXED32_MODE, "parent": list(kernel._FR13_FIXED32_PARENT),
                                   "valid_mask": hex(FX.T.HYDRA27_VALID_MASK), "active_drafts": int(FX.T.HYDRA27_ACTIVE_DRAFTS), "physical_rows": N,
                                   "schedule_expected": {k: (v if isinstance(v, (int, str, tuple, list)) else str(v)) for k, v in dict(getattr(kernel, "_FR13_FIXED32_SCHEDULE_EXPECTED", {})).items()},
                                   "geom_override": kernel._read_tree_gdn_geom_override(), "scan_align": bool(kernel.scan_align_on())},
            "geometry": {"H": H, "HV": HV, "K": K, "V": V, "physical_rows": N, "bank_rows": ROWS},
            "bank_off16_note": "legacy argument of launch_tree_gdn_replay_all_layers; fixed32 replay binds the preseeded bank tuple (banks_list) and does not read it as a pointer table",
            "component_storage": {"alias_classes": ALIAS_CLASSES, "alias_width": ALIAS_WIDTH, "ssi_groups": SSI_GROUPS, "page_elems_bf16": PAGE_ELEMS_BF16, "rows_per_page": ROWS, "capacity": CAPACITY,
                                  "conv_bank_shape": [ROWS, CONV_C, CONV_L], "ssm_bank_shape": [ROWS, HV, V, K], "row_assignment": {"warm_alias_rows": list(WARM_ROWS), "running_rows_by_rank": [run_row(16 * r) for r in range(3)],
                                  "scratch_rows_by_rank": [scratch_row(16 * r) for r in range(3)], "control_rows_by_rank": [list(control_rows(16 * r)) for r in range(3)]},
                                  "source_rows_per_batch": SOURCE_ROWS, "note": "setup fidelity for the production boot lifecycle; convolution numerics are OUT OF SCOPE"},
            "no_model_boot": True, "no_timing_claims": True,
            "conv_boundary": "conv warm/setup executed through the public production boot lifecycle (register x3 -> committer preseed -> conv pregather preseed -> selfcheck -> lease audit -> warm); NO conv qualification cases; conv numerics out of scope"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-root", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--expected-observations", required=True)
    ap.add_argument("--block", default="calibration"); ap.add_argument("--out", required=True); ap.add_argument("--process-tag", required=True); ap.add_argument("--image-id", required=True)
    ap.add_argument("--expected-fixture-manifest-sha256", required=True); ap.add_argument("--expected-observations-sha256", required=True)
    ap.add_argument("--policy", required=True); ap.add_argument("--expected-policy-sha256", required=True)
    ap.add_argument("--gate-snapshot", required=True, help="launch-time copy of GATE-Q1.2b.json: the EXTERNAL approval record (policy bytes stay immutable)")
    ap.add_argument("--tensor-store", required=True, help="shared content-addressed tensor directory (both processes)")
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--negatives", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--init-only", action="store_true", help="initialization smoke: identity refusal + full production boot lifecycle + receipt, then EXIT before any fixture phase")
    args = ap.parse_args()
    t0 = time.time(); refusals = []
    if os.path.exists(os.path.join(args.out, "result.json")):
        refusals.append("output already holds a result.json")
    kernel_path = os.path.join(ROOT, "src", "lumo_flywheel_serving", "fr10_gdn_tree_kernel.py")
    if sha256_file(kernel_path) != EXPECTED_KERNEL_SHA:
        refusals.append("kernel module sha drift")
    if sha256_file(os.path.join(ROOT, "scripts", "fr13_fixed32_topology.py")) != EXPECTED_TOPOLOGY_SHA:
        refusals.append("topology sha drift")
    for k, v in REQUIRED_ENV.items():
        if os.environ.get(k) != v:
            refusals.append(f"env {k}={os.environ.get(k)!r} != required {v!r}")
    if sha256_file(args.manifest) != args.expected_fixture_manifest_sha256:
        refusals.append("fixture manifest sha mismatch")
    if sha256_file(args.expected_observations) != args.expected_observations_sha256:
        refusals.append("expected-observations sha mismatch")
    approval = gate_approval(args.gate_snapshot)
    gate_doc = json.load(open(args.gate_snapshot))
    pol = PE21.load_policy_verified_v2_1(args.policy, args.expected_policy_sha256, approval)
    auth = authorize_scope(gate_doc, pol, args)
    refusals += auth["problems"]
    manifest = json.load(open(args.manifest)); obs = json.load(open(args.expected_observations))
    bound_cases, bind_problems = PE21.bind_expected_manifest(pol, obs, args.expected_observations_sha256, args.block) if pol.get("_integrity_ok") else ({}, ["policy integrity failed"])
    if bind_problems:
        refusals.append(f"expected-observation manifest not bound: {bind_problems[:5]}")
    if pol.get("domain", {}).get("q1_2b", {}).get("fixture_manifest_sha256") != args.expected_fixture_manifest_sha256 or pol.get("domain", {}).get("q1_2b", {}).get("expected_observations_sha256") != args.expected_observations_sha256:
        refusals.append("policy v2.1 domain does not bind THIS fixture manifest / expected observations")
    if obs.get("fixture_manifest_canonical") != manifest.get("canonical_sha256_excluding_timestamp"):
        refusals.append("expected observations not built from this fixture manifest")
    block_problems, block_facts = ([], {"skipped": "init_only"}) if args.init_only else block_authorization(gate_doc, pol, manifest, args)   # v2.2: held-out block guard
    refusals += block_problems
    eligible = (args.block in SUPPORTED_BLOCKS and not block_problems and args.repeats >= 2 and args.limit == 0)
    from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as native_mod
    if sha256_file(native_mod.__file__) != EXPECTED_NATIVE_SHA:
        refusals.append("native module sha drift")
    import lumo_flywheel_serving.fr10_gdn_tree_kernel as kernel
    if kernel._FR13_FIXED32_MODE != "hydra27_fixed32" or tuple(kernel._FR13_FIXED32_PARENT) != tuple(FX.T.PHYSICAL_PARENT):
        refusals.append("kernel mode/parent differs from authority")
    opts = {n: getattr(kernel, n)() for n in ("_fr13_fixed32_committer_direct_metadata_requested", "_fr13_fixed32_committer_sticky_guard_requested", "_fr13_fixed32_committer_metadata_fusion_requested",
                                              "_fr13_fixed32_committer_knorm_ring_requested", "_fr13_fixed32_committer_gate_ring_requested", "_fr13_fixed32_committer_decay_ring_requested", "_fr13_fixed32_committer_layer_batch_requested")}
    if any(opts.values()) or kernel.scan_align_on() or kernel._read_tree_gdn_geom_override() != {"BV": 8}:
        refusals.append(f"committer/scan options differ from the receipt defaults: {opts}")
    os.makedirs(args.out, exist_ok=True)
    att = attest(args, kernel_path, native_mod, kernel); att["committer_options"] = opts; att["eligible_mode"] = eligible; att["policy_sha256"] = pol.get("_loaded_sha256"); att["policy_version"] = pol.get("version")
    att["authorization"] = auth
    att["approval"] = approval; att["expected_observations_sha256"] = args.expected_observations_sha256; att["fixture_manifest_sha256"] = args.expected_fixture_manifest_sha256; att["block"] = args.block; att["repeats"] = args.repeats
    att["block_authorization"] = block_facts
    entries = [e for e in manifest["fixtures"] if e["block"] == args.block]
    if args.limit:
        entries = entries[: args.limit]
    att["projected_archive"] = projected_archive_bytes(int(manifest["geometry"]["layers_per_fixture"]), len(entries), args.repeats)
    if refusals:
        att["refusals"] = refusals
        json.dump({"schema": SCHEMA_RESULT, "attestation": att, "refused_before_gpu_work": True, "eligible": eligible, "fixtures": []}, open(os.path.join(args.out, "result.json"), "w"), indent=1)
        print("REFUSED:", json.dumps(refusals)); sys.exit(2)
    _cpu_pages = build_alias_pages(torch.device("cpu"), 4)
    pre_boot = validate_row_policy() + validate_alias_topology(_cpu_pages["conv_banks"], _cpu_pages["ssm_banks"], lambda l: l // 16)
    del _cpu_pages
    if pre_boot:
        json.dump({"schema": SCHEMA_RESULT, "attestation": att, "refused_before_gpu_work": True, "eligible": eligible, "fixtures": [], "refusals": pre_boot}, open(os.path.join(args.out, "result.json"), "w"), indent=1)
        print("REFUSED (storage layout self-check):", pre_boot); sys.exit(2)
    boot_error = None
    try:
        be = Backend(torch.device("cuda:0"), int(manifest["geometry"]["layers_per_fixture"]))
    except Exception as e:  # noqa: BLE001  -- initialization failure is sealed as a receipt, never hidden
        boot_error = f"{type(e).__name__}: {e}"
        att["boot_error"] = boot_error
        json.dump({"schema": SCHEMA_RESULT, "attestation": att, "refused_before_gpu_work": False, "init_only": bool(args.init_only), "initialization_ok": False,
                   "eligible": eligible, "characterization_complete": False, "integrity_ok": False, "fixtures": []}, open(os.path.join(args.out, "result.json"), "w"), indent=1)
        raise
    att["preseed_contract"] = be.preseed.get("graphs", {}).get(1) if isinstance(be.preseed, dict) else None
    att["boot_sequence"] = be.boot; att["conv_commit_contract"] = _jsonable(be.conv_contract); att["conv_lease_audit"] = _jsonable(be.conv_audit)
    att["warm_counters"] = _jsonable(be.warm_counters); att["page_geometry"] = be.pages["geometry"]; att["storage_layout_selfcheck"] = validate_row_policy() + validate_alias_topology(be.conv_banks, be.banks, lambda l: l // 16) + ([] if int(be.conv_banks[0].shape[0]) == ROWS else ["device pages do not have the configured row count"])
    if att["storage_layout_selfcheck"]:
        raise RuntimeError(f"device storage layout self-check failed after boot: {att['storage_layout_selfcheck']}")
    if args.init_only:
        out = {"schema": SCHEMA_RESULT, "attestation": att, "init_only": True, "initialization_ok": True, "refused_before_gpu_work": False, "block": args.block, "repeats": args.repeats,
               "eligible": eligible, "fixtures_expected": len(entries), "fixtures_done": 0, "characterization_complete": False, "integrity_ok": None,
               "pointer_identity_at_alloc": be.identity, "fixtures": [], "process_wall_seconds_for_scheduling_only": time.time() - t0,
               "note": "INITIALIZATION SMOKE: production boot lifecycle executed and attested; no fixture reference/candidate phase, no tensors, no numerical evidence"}
        with open(os.path.join(args.out, "result.json"), "x") as f:
            json.dump(out, f, indent=1, allow_nan=False)
        print(json.dumps({"init_only": True, "initialization_ok": True, "warm_ready": be.boot["warm_ready"], "restoration_flags": be.boot["restoration_flags"], "measured_counters_zero": be.boot["measured_counters_zero"]}))
        sys.exit(0)
    out = run_process(be, args, entries, bound_cases, pol, att, eligible, t0)
    print(json.dumps({"eligible": out["eligible"], "complete": out["characterization_complete"], "integrity_ok": out["integrity_ok"], "fixtures_done": out["fixtures_done"],
                      "records": out["raw_inventory"]["records"], "tensors_written": out["tensor_store"]["written"], "bytes": out["tensor_store"]["bytes_written"]}))
    sys.exit(0 if out["integrity_ok"] else 3)


def run_process(be, args, entries, bound_cases, pol, att, eligible, t0=None):
    """Per-process orchestration after identity refusal and backend construction (CPU-testable with a stub backend)."""
    t0 = t0 or time.time()
    store = TensorStore(args.tensor_store); inv = Inventory(os.path.join(args.out, "raw_inventory.jsonl"))
    results = []
    for e in entries:
        p = os.path.join(args.fixtures_root, e["path"])
        if sha256_file(p) != e["sha256"]:
            results.append({"fixture_id": e["fixture_id"], "status": "FIXTURE_HASH_MISMATCH", "structural_ok": False}); continue
        fx = torch.load(p, map_location="cpu", weights_only=False)
        results.append(run_fixture(be, fx, e, bound_cases, args, args.out, store, inv, pol))
    complete = eligible and len(results) == len(entries) and all(r.get("status") == "DONE" for r in results)
    integrity_ok = complete and all(r.get("structural_ok") for r in results)
    out = {"schema": SCHEMA_RESULT, "attestation": att, "block": args.block, "repeats": args.repeats, "negatives": args.negatives, "eligible": eligible,
           "fixtures_expected": len(entries), "fixtures_done": sum(1 for r in results if r.get("status") == "DONE"), "characterization_complete": complete, "integrity_ok": integrity_ok,
           "pointer_identity_at_alloc": getattr(be, "identity", None), "fixtures": results,
           "raw_inventory": {"path": os.path.relpath(inv.path, args.out), "records": inv.n, "sha256": sha256_file(inv.path)},
           "tensor_store": {"path": args.tensor_store, "written": store.written, "reused": store.reused, "bytes_written": store.bytes_written},
           "process_wall_seconds_for_scheduling_only": time.time() - t0}
    with open(os.path.join(args.out, "result.json"), "x") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    return out


if __name__ == "__main__":
    main()
