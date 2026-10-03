# Joint raw audit → categorical projection: bounded connection review

**PASS for this two-expression integration delta.** Auditor SHA `9f28dd94d2fc243c70e42769573b5c33e5613ce749c95113efc0e72ff9aae992`; connected projection helper snapshot SHA `45187dfa879c79bbdcb2bffe963cce3ca913702628086c235bb98062ce3f40f4`. The projection helper/policy review belongs to the separate assigned reviewer; this note does not replace it.

AST comparison against accepted auditor `4f0703bd…` proves that only the new import and `native_categorical=Categorical.native_projection(doc)` return keyword were added. All target/raw/map/alias/span/device predicates remain identical. The new expression is evaluated after successful `MTP.audit`; an exception cannot produce a returned observation/baseline row.

Five connected CPU controls pass. They reuse retained full target/MTP synthetic raw payloads in a separate fixture directory and add the exact production `topk_operator` marker missing from the older synthetic metadata, updating its bootstrap digest and outer seal without changing any payload bytes. Call-through spies verify that the actual projection receives the identical document object supplied to the successful real MTP auditor, equal to the sealed record on disk. They do not replace either function's validation.

The positive checks all four phase bindings: before-Z first/follow are continuation records 0/1 at extent 65 with input positions 64/65; after-Z first is record 2 at extent 66, input position 65; target O2 consumes the declared Z token at position 65. Raw score hashes, exact ordered top-3 captures, operator, request identity, first→follow spine input and target-O2→after-Z input all match their audited source fields. Projection keeps candidate observation and qualification false.

A changed MTP raw decision, a changed target raw decision, and missing continuation records each refuse with **zero projection calls** and no returned row. A missing operator marker passes the raw stage but fails the projection itself, likewise returning no row. This distinguishes raw validity from the additional categorical capture requirement.

Evidence: `p0/monitor/review-response-20260927/native-joint-common-o0-projection-connection-review/` contains the exact source snapshot, AST proof, controls/log/results, separately resealed fixtures and review seal. Old fixtures and reviews are untouched. No GPU/model/container/remote action, gate change, live baseline result or qualification claim is involved.
