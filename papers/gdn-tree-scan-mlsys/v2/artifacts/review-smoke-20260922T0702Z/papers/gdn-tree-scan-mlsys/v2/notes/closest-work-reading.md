# Close-reading record

Read on 2026-09-21 from the primary arXiv HTML versions. Metadata is in citation-verification.json. The substantive comparison is written once in main.tex, Section II and its comparison table.

- [Bole v1](https://arxiv.org/html/2608.01651v1): read the recurrence derivation and finite polynomial solver (Section IV), state commitment, GPU pipeline, evaluation setup, online trace protocol, kernel measurements, and cumulative ablations (Sections V-VI).
- [TreeWY v1](https://arxiv.org/html/2608.20961v1): read the WY system and reconstruction equations, numerical checks and graph limitations (Sections III-IV), serving evaluation, and appendices A-F, particularly the separate admission and per-step-cost explanations.

Draft decisions: remove the old priority language; distinguish reference agreement from distribution preservation; avoid transplanting published speed ratios between configurations; propose a fixed-arrival cache experiment separately from a live task-quality run. No author was contacted and no external method was executed.

Follow-up: [numerical-comparison-decision.md](numerical-comparison-decision.md) records the repository's WY failure/correction chronology and the selected E7a/E7b mechanism experiments. The manuscript now explains why algebraic identities, local arithmetic agreement, and continuation agreement require distinct evidence. The core plan is P0 -> E7a -> E2/E7b -> E1, with E3 conditional; full serving and authors-system comparisons are deferred.
