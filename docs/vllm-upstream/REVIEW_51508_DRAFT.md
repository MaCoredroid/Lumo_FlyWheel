# #51508 adjudicating review comment (funded item B) — v2 (Codex amended text verbatim, 2026-09-17)
> Codex B_review.md: NO-GO on agent draft as written; GO on source-based
> replacement. B_amend.md: GO on the amended text after the probe logs were
> published. Placement: ONE ordinary review COMMENT on #51508; no pointers on
> #48475/#50021 now (no actionable request there; link back only if asked).
> No position on which PR should merge; no "absent human reviews = absent
> ownership" claim. Evidence: results/upstream/51508 (pinned commit in the link;
> evidence files byte-identical to the Codex-verified 771d9f9e8 tree, README
> corrected afterwards per Codex: C is counts [2,2]; stale probe docstring noted).
> AWAITING MARK GO. Body below the separator.

---

Source cross-check at main `80447d2765`, #48475 `f70b0ffe66`, #50021 `9a198c0f84`, and this PR `54b69f5dd5`: the [probe sources and logs](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/37de8398fb436be20dedf51dc561c2f3429c4e71/results/upstream/51508) preserve each gating kernel except its imports. They plant a live block ID before the index-tensor view, within the backing allocation.

For that injected zero-count/live-row case, main's negative-column load selects the planted state and then writes updated state. #48475 clamps to slot 0 and still updates. This PR's kernel clamp does likewise; its builder-level NULL_BLOCK_ID fill is what makes the existing guard skip the row. #50021 instead masks the load with other=0, reaches that guard, leaves state untouched and zeroes output. Thus #50021 is not clamp-only. This distinction does not select which PR should merge.

Production reachability remains unestablished: valid counts can become zero, but propagation requires prior drafts; discarded mid-prefill rows normally have none. A constructed kernel input does not establish that conjunction or end-to-end correctness. Published GB10 logs confirm these state-write outcomes; main's block-5 count-0/count-1 max|Δ| is 1.57916, while #50021 leaves it untouched (v2 verdict).

AI assistance was used.
