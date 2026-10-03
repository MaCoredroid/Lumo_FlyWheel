# LumoTree manuscript source — October 2 evening revision

Compile main.tex with pdflatex and BibTeX, or `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. This package includes every project-local input used by the verified canonical build and its bibliography. It does not require experiment inference.

This revision adds held-out numerical continuation, repeated recorded-request replay, sampler-audit and descriptive phase/allocation results. Exact sampling is claimed only for penalty-free settings with correct target probabilities and matching filters, including tie handling. Measured presence-penalty-1.0 tree runs depart from exact sampling after the first token of each step. Numerical agreement is tolerance-based, rather than bitwise or strict greedy equality. Timings compare the implemented sampling procedures, rather than equal output distributions.

The separate results supplement preserves source snapshots, sealed records, run manifests and arithmetic reducers. Original failed/inconclusive records are retained. Large full-model tensor objects and some full engine logs are outside this compact supplement; their source-bound references and original reducer are preserved. No arXiv upload or submission is performed by packaging.
