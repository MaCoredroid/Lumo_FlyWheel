# FR13 fixed32 kernel stack plus DFWD BM8 live gate

This source artifact binds the DFWD unified-attention BM8 live gate to the
fixed32 kernel production-selector stack. It makes no GPU byte-parity, speed,
SWE-Verified quality, hardware-floor, or acceptance claim.

## Source lineage

- Production-selector base: `7a0072f8b86bf25daf4ba3ca69e937d08a7049e4`
- DFWD BM8 source commit: `d4f1780b3530107d2349d9f4f2a66912f20cad06`
- Integrated code commit: `a477684375fce02ecbe0b0563600042943609900`
- Branch: `agent/fixed32-kernel-unified-bm8`

The BM8 source commit is a direct child of the production-selector base. The
integration therefore required no conflict resolution and preserves the base
patcher and launcher behavior byte-for-byte outside the BM8 changes.

## Selector contract

- FA2 qrow16 live and production selectors remain default-off and fail closed.
- DFWD five-head padding and its all-head byte A/B remain default-off.
- TAW native-precompute diagnostic and production selectors remain default-off.
- GDN path-BV diagnostic and production selectors remain default-off.
- CFWD committer layer batching remains default-off.
- B2-B4 two-launch GDN diagnostic and production selectors remain default-off.
- DFWD unified-attention BM8 is a default-off B1 diagnostic only. The internal
  BM8 selector is launcher-private.
- During the BM8 live gate, the captured stock BM16 graph result is served.
  Stock BM16 and candidate BM8 are recalled only into private outputs after the
  first measured real SWE-Verified replay. A PASS requires raw-byte equality
  for all four MTP calls and exactly four BM8 candidate dispatches.

## Verification

All checks ran with GPU visibility disabled:

```text
python3 -m py_compile <four integrated Python modules>
PASS

bash -n scripts/fr13_launch_forked_fa2_tree_server.sh
PASS

pytest -q <ten focused kernel-selector modules>
63 passed in 2.07s
```

No shared object was built, no server or campaign was launched, and no active
run output was read or modified.
