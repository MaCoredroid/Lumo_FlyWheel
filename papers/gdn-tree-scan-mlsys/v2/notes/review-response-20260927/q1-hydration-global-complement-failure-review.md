# Hydration shared-storage guard failure: independent bounded review

**The observed address overlap is confirmed. Exact global destination-span subtraction is the appropriate bounded guard repair.** This is a failure diagnosis and prospective design review, not approval of successor source or a retry.

Run: `q1-candidate-stage1-20260929T083835Z`. The terminal receipt reports driver exit 2, cleanup exit 0, owned container stopped and removed. One HTTP request was attempted, zero observations were authenticated valid, and repeat 1 was not attempted. The model became healthy after 292 seconds. There is no completed post-import O0, O1, O2, or candidate qualification result.

The sealed failed case SHA256 is `c39663103a16910a86568c2bf170a0f2721315b2ff5ac596ef5ec6f7816ad071`; its canonical seal and receipt member hash independently match. Its bound hydration-v3 source SHA256 is `c311af48af6e65316ea1da0c471f57eefd0e87422a8b030733ce3bfb6efd3cc2`, matching the reviewed local source. The KV reblocking helper is `f3a5957f2ea8f934818500d1a21e1ba087e496460575fa71c3750f52a8f08213`. Relevant source/metadata copies and exact hashes are preserved under `p0/monitor/review-response-20260927/hydration-global-complement-failure-review/`.

## Confirmed byte-address overlap

Natural O0 records show GDN layers 0/1/2 and attention layer 3 share base address `181029789696`. Both the GDN row and physical attention-page strides are **4,194,304 bytes**. Selected GDN rows are 53, 54, and 55. Attention's imported physical pages include 52 and 56; page size is 1024 tokens, with native64 export chunks distinct from physical pages.

The failing `conv_row 52` guard is `[181247893504, 181248589824)`, **696,320 bytes**. It lies entirely inside the imported page-52 K interval `[181247893504, 181249990656)`. Thus those guard bytes are legitimate attention-import destinations. Hydration-v3's row-based GDN exclusion at lines 97–106 removes selected GDN rows only; it does not subtract attention writes to the same physical storage. The guard mismatch is consistent with a false untouched-region assertion, not evidence that imported native values were wrong.

A partial-overlap case in the same recorded layout is important: `conv_row 56` contains 696,320 bytes, while the materialized portion of attention page 56 is 175 tokens. Its K writes occupy only **358,400 bytes** of that conv guard, leaving **337,920 bytes that must still be guarded**. Dropping the entire neighboring row because any portion overlaps would hide real corruption.

The failure stage is `guards`, after source-v3's final per-region and native-order digest checks. Therefore the executed source path implies those checks completed before the reported failure. This review does not claim to have independently recomputed all imported raw contents, nor that the later registry-identity check completed; it occurs after the guards and was not reached.

## Required properties of the prospective repair

- Compute the permitted-write union from the exact already-validated destination views for **all GDN and attention imports**, before any mutation. Preserve the existing rejection of colliding destinations and the exact source-object authentication. Do not substitute entire rows/pages or storage allocations for the actual written byte spans.
- For each existing guard descriptor, compare precisely its physical byte interval minus that global union. This must apply to convolution/SSM neighboring rows, null rows, KV tails, and neighboring KV blocks alike. Fully covered guards may have an empty complement, but must be reported as excluded bytes; partially covered guards retain every remaining byte.
- Use actual device/address/stride geometry. For these contiguous guard views, byte offsets into the original view must correspond exactly to physical offsets. Unsupported noncontiguous layouts should fail before mutation unless their byte ranges are explicitly derived; a bounding interval must not silently treat unwritten gaps as permitted writes.
- Keep the original full convolution-padding zero precondition before mutation. Preserve mutation latching, every final per-object readback, the complete native-order O0 digest, post-import registry identity checks, and hard process fail-stop. A failure after any attempted write remains unusable; no rollback or successful hydration is implied.
- Record original guard bytes, excluded permitted-write bytes, and compared complement bytes separately. Existing bounded guard scope remains bounded; this change does not establish whole-storage coverage.

Minimal powered controls: fully covered row 52 passes legitimate imports; partial row 56 passes legitimate writes but refuses a one-byte change in its 337,920-byte complement; adjacent nonoverlapping spans do not subtract a byte; multiple permitted intervals preserve gaps; out-of-union mutation still fails; destination overlap and nonzero-padding preconditions still refuse before writes. Include the reverse view: a guarded neighboring KV page partially occupied by selected GDN taps/SSM must retain its unwritten complement.

No GPU, container, cache, remote mutation, implementation edit, or gate change occurred in this review. This diagnosis does not reopen numerical criteria or permit a retry. The convolution-publication observer source review remains a separate pending item.
