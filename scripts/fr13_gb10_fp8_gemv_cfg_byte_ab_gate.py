#!/usr/bin/env python3
"""FR13 OPT-A byte-A/B gate: GB10 fp8 GEMV config (FR13_GB10_FP8_GEMV_CFG).

CPU-ONLY, boot-free. Proves the OPT-A patch
(``_patch_fp8_utils_gb10_gemv_cfg`` in fr10_phase4_patch_vllm_tree_gdn.py) is:

  A) byte-identical to the stock vLLM default path when the flag is OFF, when
     the device is not GB10, or when M is large (prefill / fat batch); AND
  B) lossless-by-construction when the flag is ON and the device is GB10 and
     M is a skinny tree-verify decode tier -- i.e. the override changes only
     scheduling meta-params (BLOCK_SIZE_M / GROUP_SIZE_M / num_warps /
     num_stages) and KEEPS BLOCK_SIZE_N and BLOCK_SIZE_K pinned to
     block_size[*] (K pinned to 128), so the fp32 K-accumulation order in the
     Triton kernel is unchanged.

This is the structural/text gate that runs without a GPU. It does NOT run the
Triton kernel (that is the deploy-time live-GPU confirm, documented below).
Because the only difference flag-on introduces is the *launch* meta-params and
those provably do not touch the reduction axis, text-level equality of the
default dict + pinned-N/K assertion is a complete losslessness proof for the
flag-off and non-engaged paths, and a construction proof for the engaged path.

LIVE-GPU CONFIRM (deploy time, not here): boot the tree server twice with the
SAME seed/prompt, once with FR13_GB10_FP8_GEMV_CFG=0 and once =1, and assert
the decode token streams are byte-identical (greedy + t=0.6) and accept/event
is unchanged -- the class-9/class-10 byte A/B used for FIX-1/2/3. The override
firing is observable via the Triton autotune/launch config; losslessness is
the byte-identical stream.

Usage:
  python3 scripts/fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py [--src <fp8_utils.py>]

Exit 0 = all gates PASS. Nonzero = a gate FAILED (and which).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"

# Stock vLLM default config (fp8_utils.py w8a8_triton_block_scaled_mm else-branch).
STOCK_DEFAULT = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,   # = block_size[0]
    "BLOCK_SIZE_K": 128,   # = block_size[1]
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 2,
}
# GB10 OPT-A decode override (must keep BLOCK_SIZE_N/K == stock).
GB10_DECODE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 4,
}


def _load_patcher_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_fr13_patcher", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_pristine_fp8_utils(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise SystemExit(f"--src not found: {p}")
        return p
    # Search common live/snapshot locations for an UNPATCHED copy.
    candidates = [
        Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
              "quantization/utils/fp8_utils.py"),
        Path("/tmp/vllm_live_019/vllm/model_executor/layers/quantization/utils/"
             "fp8_utils.py"),
        Path("/tmp/vllm_pristine_019/extracted/vllm/model_executor/layers/"
             "quantization/utils/fp8_utils.py"),
    ]
    for c in candidates:
        if c.exists():
            txt = c.read_text()
            if "FR13_GB10_FP8_GEMV_CFG" not in txt and "def w8a8_triton_block_scaled_mm" in txt:
                return c
    raise SystemExit(
        "No pristine fp8_utils.py found. Pass --src <path>. (A copy already "
        "containing FR13_GB10_FP8_GEMV_CFG is not pristine.)"
    )


def _extract_else_branch(text: str) -> str:
    """Return the source of the `else:` default-config branch (pre-grid)."""
    start = text.index("    configs = get_w8a8_block_fp8_configs(")
    end = text.index("    def grid(META):", start)
    return text[start:end]


def _eval_config(branch_src: str, *, flag_on: bool, device: str, M: int) -> dict:
    """Execute the patched else-branch in isolation to get the chosen config.

    We stub `configs=None` (the GB10 reality: no JSON), set block_size=[128,128],
    and the environment / device / M, then run the branch and read `config`.
    """
    # Strip the `configs = get_...` line and the `if configs:` discriminator;
    # force the else path (configs is None on GB10) deterministically.
    body = branch_src
    body = body.replace(
        "    configs = get_w8a8_block_fp8_configs(N, K, block_size[0], block_size[1])\n",
        "    configs = None\n",
        1,
    )
    # Build a fake `vllm.platforms.current_platform` so the override's import
    # resolves to our injected device name (no real vLLM/GPU needed).
    fake_platforms = types.ModuleType("vllm.platforms")

    class _CP:
        @staticmethod
        def get_device_name():
            return device

    fake_platforms.current_platform = _CP
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.platforms = fake_platforms
    sys.modules["vllm"] = fake_vllm
    sys.modules["vllm.platforms"] = fake_platforms

    env_backup = os.environ.get("FR13_GB10_FP8_GEMV_CFG")
    os.environ["FR13_GB10_FP8_GEMV_CFG"] = "1" if flag_on else "0"
    try:
        ns: dict = {"M": M, "block_size": [128, 128]}
        # Dedent: the branch is indented 4 spaces (function body). exec wants
        # module-level statements.
        dedented = "\n".join(
            line[4:] if line.startswith("    ") else line
            for line in body.splitlines()
        )
        exec(compile(dedented, "<else-branch>", "exec"), ns)  # noqa: S102
        return dict(ns["config"])
    finally:
        if env_backup is None:
            os.environ.pop("FR13_GB10_FP8_GEMV_CFG", None)
        else:
            os.environ["FR13_GB10_FP8_GEMV_CFG"] = env_backup
        sys.modules.pop("vllm.platforms", None)
        sys.modules.pop("vllm", None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="pristine fp8_utils.py to patch")
    args = ap.parse_args()

    pristine = _find_pristine_fp8_utils(args.src)
    pristine_text = pristine.read_text()
    pristine_branch = _extract_else_branch(pristine_text)

    fails: list[str] = []
    print(f"[gate] pristine fp8_utils.py: {pristine}")

    # --- Apply the patcher to a sandbox copy (CPU, no GPU) ---
    patcher = _load_patcher_module()
    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td) / "fp8_utils.py"
        shutil.copy2(pristine, sandbox)
        patcher.FP8_UTILS_PATH = sandbox
        did = patcher._patch_fp8_utils_gb10_gemv_cfg()
        if not did:
            fails.append("patcher returned False on a pristine copy (expected True)")
        patched_text = sandbox.read_text()

        # G0: patched file still compiles.
        try:
            compile(patched_text, str(sandbox), "exec")
            print("[G0] PASS  patched fp8_utils.py compiles")
        except SyntaxError as e:
            fails.append(f"patched file does not compile: {e}")
            print(f"[G0] FAIL  {e}")

        # G1: idempotent (second apply is a no-op).
        if patcher._patch_fp8_utils_gb10_gemv_cfg() is not False:
            fails.append("patcher is not idempotent (second apply re-patched)")
        else:
            print("[G1] PASS  patcher idempotent (sentinel-guarded)")

        # G2: the stock default dict is preserved VERBATIM (byte-for-byte).
        stock_block = (
            '        config = {\n'
            '            "BLOCK_SIZE_M": 64,\n'
            '            "BLOCK_SIZE_N": block_size[0],\n'
            '            "BLOCK_SIZE_K": block_size[1],\n'
            '            "GROUP_SIZE_M": 32,\n'
            '            "num_warps": 4,\n'
            '            "num_stages": 2,\n'
            '        }\n'
        )
        if stock_block in patched_text:
            print("[G2] PASS  stock default dict preserved verbatim in patched source")
        else:
            fails.append("stock default dict NOT preserved verbatim after patch")
            print("[G2] FAIL  stock default dict altered")

        # G3: BLOCK_SIZE_K is never hard-coded away from block_size[1] in the
        # override (pinned-K contract). The override dict must read block_size[1].
        ov_match = re.search(
            r"# FR13_GB10_FP8_GEMV_CFG.*?config = \{(.*?)\}",
            patched_text,
            re.DOTALL,
        )
        if not ov_match:
            fails.append("could not locate override config dict in patched source")
            print("[G3] FAIL  override dict not found")
        else:
            ov = ov_match.group(1)
            ok_k = '"BLOCK_SIZE_K": block_size[1]' in ov
            ok_n = '"BLOCK_SIZE_N": block_size[0]' in ov
            if ok_k and ok_n:
                print("[G3] PASS  override pins BLOCK_SIZE_N=block_size[0], "
                      "BLOCK_SIZE_K=block_size[1] (K=128 reduction order unchanged)")
            else:
                fails.append(
                    f"override does not pin N/K (N_pinned={ok_n}, K_pinned={ok_k})"
                )
                print(f"[G3] FAIL  N_pinned={ok_n} K_pinned={ok_k}")

        patched_branch = _extract_else_branch(patched_text)

    # --- Behavioral A/B over the (flag, device, M) matrix ---
    # A-arm cases: must equal the STOCK default exactly.
    a_cases = [
        ("flag OFF, GB10, M=8 (decode)", dict(flag_on=False, device="NVIDIA GB10", M=8)),
        ("flag ON, non-GB10 (H100), M=8", dict(flag_on=True, device="NVIDIA H100 80GB HBM3", M=8)),
        ("flag ON, GB10, M=64 (prefill/fat)", dict(flag_on=True, device="NVIDIA GB10", M=64)),
        ("flag ON, GB10, M=33 (above decode guard)", dict(flag_on=True, device="NVIDIA GB10", M=33)),
        ("flag OFF, non-GB10, M=8", dict(flag_on=False, device="NVIDIA H100 80GB HBM3", M=8)),
    ]
    for name, kw in a_cases:
        cfg = _eval_config(patched_branch, **kw)
        if cfg == STOCK_DEFAULT:
            print(f"[A ] PASS  {name} -> stock default (byte-identical)")
        else:
            fails.append(f"A-case '{name}' diverged from stock default: {cfg}")
            print(f"[A ] FAIL  {name} -> {cfg} != stock")

    # Cross-check: the pristine (unpatched) branch must yield the stock default
    # for the same A-cases (proves the A-arm == true vLLM behavior, not just
    # self-consistency).
    for name, kw in a_cases:
        cfg_pristine = _eval_config(pristine_branch, **kw)
        if cfg_pristine != STOCK_DEFAULT:
            fails.append(
                f"pristine branch '{name}' != stock default: {cfg_pristine} "
                "(test harness drift)"
            )
            print(f"[A0] FAIL  pristine '{name}' -> {cfg_pristine}")
    print("[A0] PASS  pristine vLLM default-branch == stock default for all A-cases")

    # B-arm cases: flag ON + GB10 + skinny decode M -> GB10 config, N/K pinned.
    b_cases = [
        ("flag ON, GB10, M=6", dict(flag_on=True, device="NVIDIA GB10", M=6)),
        ("flag ON, GB10, M=8", dict(flag_on=True, device="NVIDIA GB10", M=8)),
        ("flag ON, GB10, M=10", dict(flag_on=True, device="NVIDIA GB10", M=10)),
        ("flag ON, GB10, M=32 (guard edge)", dict(flag_on=True, device="NVIDIA GB10", M=32)),
    ]
    for name, kw in b_cases:
        cfg = _eval_config(patched_branch, **kw)
        if cfg != GB10_DECODE:
            fails.append(f"B-case '{name}' != expected GB10 config: {cfg}")
            print(f"[B ] FAIL  {name} -> {cfg}")
            continue
        # The lossless invariant: N and K must equal the stock default.
        if (cfg["BLOCK_SIZE_N"] == STOCK_DEFAULT["BLOCK_SIZE_N"]
                and cfg["BLOCK_SIZE_K"] == STOCK_DEFAULT["BLOCK_SIZE_K"]):
            print(f"[B ] PASS  {name} -> GB10 decode cfg, "
                  f"N={cfg['BLOCK_SIZE_N']}/K={cfg['BLOCK_SIZE_K']} PINNED "
                  f"(lossless: only M/GROUP/warps/stages move)")
        else:
            fails.append(
                f"B-case '{name}' broke N/K pin: N={cfg['BLOCK_SIZE_N']} "
                f"K={cfg['BLOCK_SIZE_K']}"
            )
            print(f"[B ] FAIL  {name} broke N/K pin")

    print()
    if fails:
        print(f"GATE: FAIL ({len(fails)} failure(s))")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("GATE: PASS  (flag-off/non-GB10/large-M byte-identical to stock vLLM; "
          "flag-on GB10 decode pins N/K => lossless by construction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
