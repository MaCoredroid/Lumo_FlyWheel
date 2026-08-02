# FR13 fixed32 next kernel stack integration

This artifact binds a coherent, default-off source integration on top of the
current B1 live-gate rejection tree. It adds the TAW and GDN production
selectors, the diagnostic unified-attention BM8 candidate, the ABI-safe FA2
qrow16 source, and the combined B2-B4 two-launch wide-BV GDN route.

It contains no new GPU, Docker, SWE-Verified, byte-parity, timing,
acceptance, or hardware-floor result.

## Lineage

- Live-gate base: `850355982cf747d2a960e7dae5f769edb660d772`
- Integrated source tip: `1f5b63c162b82f7aca29b09a1af28fd71b2ba32d`
- Branch: `agent/fixed32-kernel-next-stack`
- Production selector source: `39907d3dc10808b7d5b610483576e953467626a4`
- Unified BM8 source: `a64b883504cd8aa600942f130cad51b4d0897335`
- Qrow16 source/default-off commits:
  `c938ef2a9ca54915035bc258107dac4cdb7cab62`,
  `ff1937f1b6f7be93e6783780ce60c3cd346d2a1a`
- Batched wide-BV source/artifacts:
  `df7494658ce160411ab7dfadffd6394895e0c61a`,
  `d509384639eedea391b55f46cebbe0c75ae173e2`,
  `86f72da79bffac34e68cbfabda2bb13b66ae2a21`

The only cherry-pick conflict was in the path-BV test helper. The resolution
retains both the scalar tensor raw-byte comparator added by
`cc4f420bdcf37aca39c063ec684ef739208fa92c` and the production resolver from
the selector source.

## Safety state

All newly integrated selectors remain default-off:

- TAW diagnostic and production default to `0`.
- Qrow16 live diagnostic and production default to `0`; the B1 gate wrapper
  also defaults qrow16 to `0`.
- Unified-attention BM8 is diagnostic-only and defaults to `0`.
- B1 path-BV diagnostic and production selectors default to an empty value.
- Batched GDN diagnostic and production default to `0`; both wide-BV
  selectors default to an empty value.

The real B1 BV64 result from base commit `850355982` remains a rejection:
`REJECTED_BYTE_MISMATCH` on the `export` surface, with stock restored and no
candidate output served. It emitted no PASS record, so it cannot satisfy the
source-bound production selector and BV64 remains ineligible for B1
production.

The combined batched wide-BV route has source evidence only. Its formal
diagnostic remains an eager, metrics-enabled, real SWE-Verified exact4 B4 run
with `MAX_NUM_SEQS=4`. It serves stock bytes and requires a source-bound v2
PASS across all 48 layers before production can arm. No such PASS is included
here, so B4 production remains unarmed and no B4 timing claim is valid.

## Verification

The broad verification command used `CUDA_VISIBLE_DEVICES=''`; no GPU or
Docker command was issued.

```text
Focused selector, qrow16, BM8, and wide-BV suite:
89 passed in 2.45s

Fixed32 source-relevant broad suite:
590 passed, 8 skipped, 10 deselected in 12.37s

bash syntax: PASS
Python compilation: PASS
Ruff on merged kernel and candidate tests: PASS
```

The ten deselections are the established isolated-worktree `.venv` and local
SWE-Verified dataset-cache prerequisite cases. The eight skips are
CUDA/Triton-dependent tests.
