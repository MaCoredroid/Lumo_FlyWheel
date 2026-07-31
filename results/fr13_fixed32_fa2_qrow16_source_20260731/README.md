# Fixed32 FA2 query-row tile candidate

This experimental, not-deploy-ready branch adds a build-time opt-in FA2
geometry for the fixed32 B1 tree call:
`kBlockM=16, kNWarps=1` instead of `kBlockM=64, kNWarps=4`.

At B1, 32 query rows and 24 heads currently launch 24 CTAs on the 48-SM
GB10. The candidate launches two query CTAs per head, or 48 CTAs. Unlike the
rejected split-K candidate, each CTA owns complete query rows, reads every K
tile in the existing order, and writes its final rows directly. There is no
cross-CTA floating-point reduction or combine kernel.

The one-warp traits cannot instantiate FA2's split-K combine kernel, which is
hard-coded for 128 threads. The candidate therefore calls the launcher with a
compile-time `AllowSplit=false`; the runtime dispatch also requires
`num_splits <= 1`. This compile-time flag discards the combine block; FA2's
`BOOL_SWITCH` still compiles a `Split=true` main-kernel variant, but the runtime
predicate prevents it from executing. Stock four-warp calls retain the default
split-capable path.

The runtime guard requires `params.b == 1`. At B4, the stock geometry already
launches 96 CTAs per layer across 48 SMs; query splitting would only raise that
to 192 CTAs while rereading K/V. B4 therefore stays on the stock path.

The candidate is further restricted to the observed production signature:
`d=256`, 24 heads, full-window attention, no ALiBi, and no appended KV. Rounded
head dimensions and local/append variants remain on the stock path.

The current and candidate mappings assign each real row to the same
warp-local MMA row/lane. This is a strong construction argument, not a GPU
byte-equivalence result: changing template geometry can still change generated
instructions. A bytewise output/LSE A/B remains mandatory before any real-task
campaign.

With one warp, each paged-KV copy thread owns 16 rows. The dispatch statically
asserts that layout, which is licensed by FA2's existing public-API requirement
that paged-KV page sizes are divisible by 16.

The performance tradeoff is explicit. The candidate removes 32 masked query
rows per head and fills all SMs, but two query CTAs reread K/V. The expected
`6-12 ms/event` saving is an estimate against the real B1 FA2 attribution of
`24.7086 ms/event`; a long-context bandwidth regime can reduce or reverse it.

The remaining conv source-stage copy is not competitive: the real trace shows
48 graph memcpys of 737,280 bytes totaling `0.0897 ms/event`.

No GPU, synthetic performance probe, or acceptance run was used to create this
candidate. See `diagnostic.json` for provenance and required gates.
