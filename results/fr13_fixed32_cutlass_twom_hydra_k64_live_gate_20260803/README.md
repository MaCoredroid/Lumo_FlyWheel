# FR13 fixed32 CUTLASS two-M Hydra27 K64 live gate

Status: `QUALIFIED_BYTE_EXACT`.

This real SWE-Verified exact4 B4 diagnostic qualified the fixed32
`identity_twom_b4` route for Hydra27 at K64/root1. The diagnostic always
served the stock result and collected no timing samples.

All four canonical tasks reached terminal evaluation: two resolved and two
failed. All four model traces closed with a nonempty final result. The kernel
comparator exercised all five audited target projection shapes for 320 calls
at physical M=128. Every BF16 output byte matched stock: zero mismatching
comparisons and zero differing bytes.

The stock-served real trajectory recorded 8,079 speculative events, with
4.778562 accepted drafts and 5.785122 generation tokens per event. This is
exact4 topology/trajectory telemetry, not a candidate-caused acceptance delta
or a timing result.

The launch and end runtime manifests were identical, as were the launch and
end external manifests. The source was pinned at `30f8437d2`; the candidate
binary was 117,488,608 bytes with SHA-256
`c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29`.

No task identifiers, prompts, responses, patches, raw logs, environment,
process identifiers, or container identifiers are published here.

## Scope

Together with the separately published Tail23 PASS on the identical source
and binary, this result authorizes paired K64/root1 full-wall timing. It is not
a timing, TPS, one-sided-U95, exact16, or hardware-floor acceptance result.
