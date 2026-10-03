# Convolution witness v2: identity closure

**PASS for the bounded F1 source repair.** `q1_conv_publication_witness_v2.py` SHA256 `d9dac5bf159bef54694e3af34ea847e7e1fb951d26f5e46c14c8cf3e4bc3b7ec` closes the strict-identity gap in the v1 review. No remaining issue found in this delta.

`span()` now accepts canonical `cpu` or indexed `cuda:N` labels; `plan()` requires every repeated object ID across banks and sources to retain all identity fields. Exact legitimate same-object aliases and separate tensor objects over the same accepted bank view remain valid. The only other change is the prospective schema version.

Independent standard-library controls pass four valid alias/device cases and refuse 11 malformed labels, contradictory IDs within bank classes, within sources, and across families, plus same-pointer/different-size metadata. The three original malformed cases now refuse. AST equality confirms the source map, capture, raw byte oracle, and O1 binding are unchanged; the already-passing full saved-byte oracle was not unnecessarily rerun. Controls and the exact source are saved in `p0/monitor/review-response-20260927/conv-publication-witness-v2-identity-review/`.

This accepts the helper repair only. Future hooks/job/raw-auditor wiring must select and bind these v2 bytes and schema. No GPU operation, gate, launch, numerical criterion change, or full-model qualification follows from this review. The separate root hydration-H4 result is unaffected.
