# #55506 confirmation of 5f71d6f on GB10 (item I follow-up) — v2 (Codex replacement verbatim: dropped "this looks inert"; ordinary PR comment; GO; AWAITING MARK GO)
> Karl0007 fixed both findings in 5f71d6f using our test as the acceptance run (his run: sm_80).
> Our GB10 confirmation (agent I2): original file 8 pass / 1 fail (old C8 by design); his updated
> file 9 pass; padding probe track mode shows padded rows constant [1,2,3]; compute-sanitizer
> memcheck 0 errors; C2 real rows identical to a28e902. Precision note: padded rows now resolve to
> table row 0 = request slot 0's block ids, not block id 0 (the pre-PR null block); padded rows carry
> zero query tokens (model_runner.py:1341) so likely inert. Evidence: results/upstream/55506
> (CONFIRM_5f71d6f.md, I_55506_evidence_v2.tar.gz logs 70/80/90). Purpose: second-architecture
> confirmation + one precision note; ≤90 words; no merge position.

---

Confirmed on GB10 (sm_121) at `5f71d6f`: updated file 9 passed; original file 8 passed/1 failed because old C8 expects the former return type. The standalone padding probe reports 0 compute-sanitizer errors, with real-row indices unchanged from `a28e902`. One precision note: padded rows now use request slot 0's block IDs, not null block 0 as the pre-PR gathered path did. Downstream KDA behavior was not tested. [Evidence](https://github.com/MaCoredroid/Lumo_FlyWheel/tree/7b03ad8fdc12e506ea38caaafe4dc75005ce3237/results/upstream/55506). AI assistance was used.
