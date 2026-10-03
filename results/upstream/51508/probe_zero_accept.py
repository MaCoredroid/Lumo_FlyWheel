"""Model-free discriminating probe for the GDN zero-accept spec row defect.

Loads the fused_sigmoid_gating kernel from four refs as standalone modules
(only vllm dep, `from vllm.triton_utils import tl, triton`, is rewritten to
plain triton) and asks one question of each:

  for a spec-decode row whose num_accepted_tokens == 0, but whose
  ssm_state_indices row still holds LIVE block ids, does the kernel
  (a) read a state at all, and (b) WRITE BACK (advance) the recurrent state?

Case Z  raw 0 reaches the kernel, row still live   <- what main/#48475/#50021 see
Case N  row nulled to NULL_BLOCK_ID(0), count 1    <- what #51508's builder emits
Case C  healthy control, count 2                   <- must be identical everywhere
"""

import importlib.util
import sys

import torch

VARIANTS = ["main", "pr48475", "pr50021", "pr51508"]
HERE = "/home/mark/shared/tmp-scratch/B_probe"

N, TPS, NCOL = 2, 2, 4  # 2 seqs, 2 tokens each, num_spec+1 == 4 columns
H = HV = 1
K = V = 16
NUM_BLOCKS = 16
SENTINEL_BEFORE = 7  # live block id planted immediately BEFORE the index tensor
ROW0 = [5, 6, 7, 8]
ROW1 = [9, 10, 11, 12]


def load(name):
    path = f"{HERE}/variant_{name}.py"
    spec = importlib.util.spec_from_file_location(f"v_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"v_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod.fused_sigmoid_gating_delta_rule_update


def make_inputs(dev, dtype):
    g = torch.Generator(device=dev).manual_seed(1234)
    total = N * TPS
    return dict(
        A_log=torch.randn(HV, device=dev, dtype=dtype, generator=g),
        a=torch.randn(1, total, HV, device=dev, dtype=dtype, generator=g),
        b=torch.randn(1, total, HV, device=dev, dtype=dtype, generator=g),
        dt_bias=torch.randn(HV, device=dev, dtype=dtype, generator=g),
        q=torch.randn(1, total, H, K, device=dev, dtype=dtype, generator=g),
        k=torch.randn(1, total, H, K, device=dev, dtype=dtype, generator=g),
        v=torch.randn(1, total, HV, V, device=dev, dtype=dtype, generator=g),
        cu_seqlens=torch.tensor([0, TPS, 2 * TPS], dtype=torch.int32, device=dev),
    )


def run_case(fn, case, dev, dtype):
    """Returns (status, changed_blocks_row0, changed_blocks_row1, initstate_read_block)."""
    # Index tensor as a view into a flat buffer, so the element at offset -1
    # is a KNOWN live block id (7) rather than whatever the allocator left
    # there. This removes the allocation-layout dependence from the probe.
    flat = torch.empty(1 + N * NCOL, dtype=torch.int32, device=dev)
    flat[0] = SENTINEL_BEFORE
    idx = flat[1:].view(N, NCOL)
    if case == "N":
        idx[0] = 0  # NULL_BLOCK_ID row, as #51508's builder emits
    else:
        idx[0] = torch.tensor(ROW0, dtype=torch.int32, device=dev)
    idx[1] = torch.tensor(ROW1, dtype=torch.int32, device=dev)
    assert idx.stride(0) == NCOL and idx.is_contiguous()

    nacc = {"Z": [0, 2], "N": [1, 2], "C": [2, 2]}[case]
    nacc_t = torch.tensor(nacc, dtype=torch.int32, device=dev)

    # Each block gets a unique constant so a write is detectable AND we can
    # tell which block was used as the initial state.
    state = torch.zeros(NUM_BLOCKS, HV, V, K, device=dev, dtype=dtype)
    for bl in range(NUM_BLOCKS):
        state[bl] = float(bl) + 0.5
    before = state.clone()

    ins = make_inputs(dev, dtype)
    try:
        fn(
            A_log=ins["A_log"],
            a=ins["a"],
            b=ins["b"],
            dt_bias=ins["dt_bias"],
            q=ins["q"],
            k=ins["k"],
            v=ins["v"],
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=ins["cu_seqlens"],
            ssm_state_indices=idx,
            num_accepted_tokens=nacc_t,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        return (f"EXC {type(exc).__name__}: {str(exc)[:90]}", None, None, None)

    ch0 = [bl for bl in ROW0 if not torch.equal(before[bl], state[bl])]
    ch1 = [bl for bl in ROW1 if not torch.equal(before[bl], state[bl])]
    other = [
        bl
        for bl in range(NUM_BLOCKS)
        if bl not in ROW0 + ROW1 and not torch.equal(before[bl], state[bl])
    ]
    return ("ok", ch0, ch1, other)


def main():
    dev, dtype = "cuda", torch.float32
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
    print(f"row0 blocks={ROW0} row1 blocks={ROW1} planted-before-tensor={SENTINEL_BEFORE}")
    print()
    hdr = f"{'variant':10s} {'case':5s} {'status':22s} {'row0(stale) wrote':22s} {'row1(live) wrote':20s} other"
    print(hdr)
    print("-" * len(hdr))
    for name in VARIANTS:
        fn = load(name)
        for case in ("C", "Z", "N"):
            st, c0, c1, oth = run_case(fn, case, dev, dtype)
            print(
                f"{name:10s} {case:5s} {st:22s} {str(c0):22s} {str(c1):20s} {oth}"
            )
        print()


if __name__ == "__main__":
    main()
