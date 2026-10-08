# Data comment for #54928 (DFlash2 greedy divergence on Qwen3.8 GDN) — v1

> **STATUS 2026-09-10: RETIRED — DO NOT POST (Codex NO-GO, Claude agrees).**
> The thread's failure is a greedy kernel-defect class (K=1, eager, prefix
> caching OFF). Our three contracts are adjacent observations, not data on
> the cause: the chunk-alignment result needs prefix caching; the Vol III
> oracle-floor method is for sampled trees and would read as loosening a
> greedy-exactness requirement; 1.19e-7 state error is not byte identity;
> and grvwrk asked for a test module, which our stack-bound harness is not.
> Posting would be advice in a bug thread = advertising. The #54928
> diagnostics get cited inside the #54080 addendum as others' observations.

> v1 (2026-09-10). Results-only, public-sourced (Vols II/III/IV/VII), no
> code promise, answers grvwrk's open question honestly. Placement: one
> general issue comment on #54928. Pending: Codex red-team, then Mark
> reviews and posts. Do not paste the header.
> Public body: ~150 words.

---

Data from a different stack, same model family: Qwen3.6-27B FP8 (GDN
hybrid) on a GB10, chain and tree spec decode, out-of-tree. Three things
that held under repeated measurement, in case they narrow the search:

1. A byte-exact recurrent *state* carry does not guarantee a byte-exact
   *output*. We reproduced the state to 1.19e-7 and still diverged
   downstream: a correct state seeded a different kernel path whose drift
   amplified. ([Vol VII](https://macoredroid.github.io/Lumo_FlyWheel/keep-or-replay.html))
2. Bit-exact GDN prefix-cache restore needs checkpoint positions that are
   multiples of the FLA chunk size (64); an 816-token checkpoint (16×51) is
   not 64-aligned and cannot be bit-exact even in principle. Same class as
   #54076 / #53798. ([Vol IV](https://macoredroid.github.io/Lumo_FlyWheel/gdn-prefix-cache.html))
3. Instead of demanding bit identity, we score clear-margin greedy flips
   against the native oracle's own floor: 13.09% (tree) vs 12.90% (native).
   ([Vol III](https://macoredroid.github.io/Lumo_FlyWheel/gdn-tree-scan.html))

@grvwrk our parity harness is bound to our own serving stack, so it does not
transfer as a test module; the three contracts above do.
