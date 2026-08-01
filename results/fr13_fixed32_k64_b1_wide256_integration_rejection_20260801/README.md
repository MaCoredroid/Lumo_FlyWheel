# K64 B1 wide256 integration rejection

Status: **no production candidate; no K64 route or timing pair prepared**.

The audit started from `c3ee2fece6daa17927ec216ff0135c5cf3ebb1e0`
and evaluated the newest stock-symbol-exact wide256 B1 candidate before changing
the Tail23/Hydra27 physical-32 K64 route. No candidate met the required
raw-byte-equality contract, so this branch deliberately contains no runtime,
production, B4, GDN, overlap, or timing integration.

The older forced-Stream-K binary preserves all six stock symbols and their
resources, but its wide256 kernel uses eight bytes of stack and all 240 observed
real-B1 comparisons differed from stock. The newer data-parallel `256x32x128`
binary has zero stack/local memory and preserves all six stock records, but all
256 observed comparisons differed. Its call bound was exhausted before the
required packed-QKV shape `(32,14336,5120)`, its formal reducer did not run, and
its checked-in checksum inventory names a `build.log` that is absent from the
source tree. Each condition is fail-closed; the byte mismatches alone are
decisive.

The `10.923627 ms/event` optimistic recovery is roofline analysis only. It is
not a measurement and cannot be claimed as realizable without a qualified
candidate.

## Target held closed

- K64: `ROOT=1`, `K=65536`, 32 physical rows root-inclusive, 31 physical drafts.
- Tail23: `tail6_fixed32`, 23 logical drafts, mask `0x7a9ce7ff`.
- Hydra27: `hydra27_fixed32`, 27 logical drafts, mask `0x7abdffff`.
- Block map: `/workspace/scripts/fr13_dvk_subset_blocks.json`, SHA-256
  `85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff`.
- Required projection shapes are recorded in `rejection.json`; all five must be
  observed with zero differing bytes before timing.

## Reduced provenance

- Forced Stream-K build: commit `00da932214b4558464a2fd12d517cfb69cf76174`;
  reduced B1 rejection: commit `83b5e554a206ad0c9327746de638a82cd858fde7`.
- Data-parallel source: branch `agent/fixed32-wide256-dp-exact-fb35`,
  implementation commit `59bb74f99267192071d8ab4bc2d345f05625a50f`;
  reduced B1 rejection: commit `bd50c04089e781f1300408fa20c93f3682b6a4de`.

Only aggregate mismatch counts and candidate identity are republished here; no
per-request or operational data is included.

## CPU-only audit

No GPU or Docker command was run for this integration audit. Candidate identity
and static metadata were rechecked with:

```bash
sha256sum /home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_b1_stock_symbol_exact_gate_ready.abi3.so
sha256sum /home/mark/fr13_streamk_build/bin/_C_stable_libtorch.wide256_dataparallel_b1_gate_ready.abi3.so
stat -c '%a %s %n' /home/mark/fr13_streamk_build/bin/_C_stable_libtorch.streamk_force_wide256_b1_stock_symbol_exact_gate_ready.abi3.so
stat -c '%a %s %n' /home/mark/fr13_streamk_build/bin/_C_stable_libtorch.wide256_dataparallel_b1_gate_ready.abi3.so
readelf -d /home/mark/fr13_streamk_build/bin/_C_stable_libtorch.wide256_dataparallel_b1_gate_ready.abi3.so
cuobjdump --dump-resource-usage /home/mark/fr13_streamk_build/bin/_C_stable_libtorch.wide256_dataparallel_b1_gate_ready.abi3.so
```

The next valid action is a new pinned wide256 mechanism with a complete build
inventory, followed by a fresh authenticated B1 byte gate that covers all five
required K64 shapes. Timing remains forbidden until that gate is byte exact.
