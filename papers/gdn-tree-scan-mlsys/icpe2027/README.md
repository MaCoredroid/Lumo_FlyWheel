# ICPE 2027 research-track submission draft

Started 4 October 2026 from the v3 manuscript (`../v2`, commit 18eae0c55) and the penalty-fix records (`../v2/results/claude-penfix-20261003/`). Build: `latexmk -pdf main.tex` (acmart sigconf, `anonymous,review`). Current state: 13 pages total, body ends on page 9 (limit: 10 pages excluding references and appendices), zero LaTeX warnings, no author-identifying strings in the sources or rendered text.

**Framing.** The verifier makes tree speculation possible on a recurrent-hybrid model at the cost of a tuned chain-speculation stack; the measurement and audit methodology is the second contribution; parity with SGLang's chain configuration is the cost of hosting tree verification, not a shortfall.

**Decisions still open (author).**
- Title (current: "Hosting Tree Speculation on a Recurrent-Hybrid Model: Verifier Design, Audit Methodology, and Serving Evidence"); must differ from the arXiv title.
- System name macro `\sys` (current placeholder: TreeHost). One line in `main.tex`.
- Anonymized artifact: an anonymous repository holding the paper source, `results/claude-penfix-20261003/`, the reducers and the patcher diff with names and host paths scrubbed; the link goes through the submission system. Reference [1] (`anon2026artifact`) is the placeholder.
- Optional batch 2–4 replay sweep on GB10 (about one GPU day) to answer the "batch 1 only" threat.

**Dates (AoE).** Abstract 9 Nov 2026, paper 16 Nov 2026, notification 25 Jan 2027, camera-ready 12 Mar 2027; conference 24–28 May 2027, Gothenburg. Artifact Evaluation and Emerging Research tracks: details TBA.

**What moved to the appendix.** Exploratory replay variants, phase timers and drafting-pass sensitivity (pre-correction sampler); recurrent agreement and isolated GDN costs under distinct references; Algorithm 2; the full-model protocol; serving settings; task-level outcomes.

**Not cited.** The arXiv preprint (same content, different title), the earlier public volumes, and the named artifact tag.
