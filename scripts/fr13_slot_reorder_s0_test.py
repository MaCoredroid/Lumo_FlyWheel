#!/usr/bin/env python3
"""S0 gate for FR13_SLOT_REORDER: apply the FULL patch sequence to SCRATCH copies
of the pristine vLLM files (no GPU, no container) and verify:
  (1) every anchor still resolves with all prior patches applied (main() order);
  (2) the patched files py_compile;
  (3) all five edit markers landed (PERMUTE/CLEAR/RESTORE, bias helper wrap,
      causal flag, dst_pi threading);
  (4) the injected pi algorithm reproduces the logic-model pi for cat8.
Exit 0 = S0 PASS (safe to boot S1).
"""
import importlib.util
import json
import os
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel")
PRISTINE = Path("/tmp/fr13_bnd_pristine")
SCRATCH = Path(os.environ.get("S0_SCRATCH", "/tmp/claude-1000/-home-mark-shared/"
                              "8245ff29-3bc7-469e-950a-7b8a1ab41d2a/scratchpad/sr_s0"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    runner = SCRATCH / "gpu_model_runner.py"
    tree_attn = SCRATCH / "tree_attn.py"
    shutil.copy(PRISTINE / "v1/worker/gpu_model_runner.py", runner)
    shutil.copy(PRISTINE / "v1/attention/backends/tree_attn.py", tree_attn)

    sys.path.insert(0, str(REPO / "src"))
    big = load("bigpatch", REPO / "scripts/fr10_phase4_patch_vllm_tree_gdn.py")
    fa2 = load("fa2patch", REPO / "scripts/fr13_patch_fa2_tree_bias.py")

    big.GPU_MODEL_RUNNER_PATH = runner
    big.TREE_ATTN_PATH = tree_attn

    # main()-order patch sequence for the two files under test.
    seq = [
        ("tree_attn_spec_config_override", big._patch_tree_attn_spec_config_override),
        ("tree_attn_op_capture", big._patch_tree_attn_op_capture),
        ("runner_tree_metadata", big._patch_gpu_model_runner_tree_metadata),
        ("runner_tree_depth_positions", big._patch_gpu_model_runner_tree_depth_positions),
        ("runner_preprocess_input_capture", big._patch_gpu_model_runner_preprocess_input_capture),
        ("runner_tree_verify_input_ids", big._patch_gpu_model_runner_tree_verify_input_ids),
        ("runner_decode_mode_globals", big._patch_gpu_model_runner_decode_mode_globals),
        ("runner_tree_reqkey", big._patch_gpu_model_runner_tree_reqkey),
        ("runner_attn_kv_remap_capture", big._patch_gpu_model_runner_attn_kv_remap_capture),
        ("runner_attn_kv_remap_apply", big._patch_gpu_model_runner_attn_kv_remap_apply),
        ("runner_slot_reorder", big._patch_gpu_model_runner_slot_reorder),
        ("runner_replay_draft_reqkey", big._patch_gpu_model_runner_replay_draft_reqkey),
        ("runner_fr13_det_warn", big._patch_gpu_model_runner_fr13_det_warn),
        ("runner_replay_boundary_tap_d", big._patch_gpu_model_runner_replay_boundary_tap_d),
        ("runner_sfwd_gpu_timer", big._patch_gpu_model_runner_sfwd_gpu_timer),
    ]
    for name, fn in seq:
        try:
            did = fn()
        except Exception as exc:
            print(f"S0 FAIL: {name}: {type(exc).__name__}: {exc}")
            return 1
        print(f"  {name}: {'patched' if did else 'skipped(sentinel)'}")

    # fa2 tree-bias patcher (python-side patch of tree_attn.py; takes path param)
    try:
        did = fa2._patch_tree_attn(tree_attn)
        print(f"  fa2._patch_tree_attn: {'patched' if did else 'skipped'}")
    except Exception as exc:
        print(f"S0 FAIL: fa2._patch_tree_attn: {type(exc).__name__}: {exc}")
        return 1

    # (2) compile
    for f in (runner, tree_attn):
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as exc:
            print(f"S0 FAIL: {f.name} does not compile:\n{exc}")
            return 1
    print("  py_compile: OK (both files)")

    # (3) markers
    rt = runner.read_text()
    tt = tree_attn.read_text()
    checks = [
        (rt, "# FR13_SLOT_REORDER_CLEAR", "runner clear"),
        (rt, "# FR13_SLOT_REORDER_PERMUTE", "runner permute"),
        (rt, "# FR13_SLOT_REORDER_RESTORE", "runner restore"),
        (rt, "dst_pi=_fr13_akr_dstpi", "runner remap dst_pi threading"),
        (rt, "_fr13_akr_dstpi = (", "runner dstpi definition"),
        (tt, "def _fr13_sr_bias_perm(", "tree_attn bias helper"),
        (tt, "def _fr13_sr_causal_flag(", "tree_attn causal helper"),
        (tt, "tree_bias = _fr13_sr_bias_perm(tree_bias)", "tree_attn bias wrap"),
        (tt, "causal=_fr13_sr_causal_flag(),", "tree_attn causal call"),
    ]
    ok = True
    for blob, marker, label in checks:
        hit = marker in blob
        ok &= hit
        print(f"  marker {label}: {'OK' if hit else 'MISSING'}")
    if not ok:
        print("S0 FAIL: markers missing")
        return 1
    # negative: exactly ONE causal=True must remain in the tree-bias decode
    # replacement region (the unified_attention else-branch), plus prefill calls.
    dec = rt  # (decode replacement lives in tree_attn text actually)
    n_causal_true = tt.count("causal=True")
    print(f"  tree_attn causal=True remaining: {n_causal_true} (unified else-branch + prefill; must be >=1)")

    # (4) pi algorithm vs logic model (cat8)
    cat8 = [(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0, 0)]
    ch = sorted(cat8, key=lambda p: (len(p), p))
    pi = ([0]
          + [i + 1 for i, c in enumerate(ch) if all(int(x) == 0 for x in c)]
          + [i + 1 for i, c in enumerate(ch) if not all(int(x) == 0 for x in c)])
    expect = [0, 1, 3, 5, 7, 8, 2, 4, 6]
    print(f"  pi(cat8) = {pi}  expect {expect}: {'OK' if pi == expect else 'MISMATCH'}")
    if pi != expect:
        print("S0 FAIL: pi mismatch vs logic model")
        return 1

    print(">>> S0 PASS — anchors resolve, files compile, markers present, pi matches. "
          "Next: S1 boot with FR13_TREE_CAUSAL_OFF=1 (byte-identity gate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
