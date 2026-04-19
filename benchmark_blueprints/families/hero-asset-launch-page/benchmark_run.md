# Benchmark Run: Hero Asset Launch Page

## Run Metadata
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Agent id: `019da331-babe-7080-aee9-a05a0bb09b2e`
- Run context: family package only under this directory
- Files available to solver:
  - `task_spec.md`
  - `evaluator_contract.md`
  - `codex/config.toml`
  - `skills/launch-hero-brief-reconcile/SKILL.md`

## Actual Solver Attempt
The child agent attempted the task from the family package and produced:
- A detailed reconciliation workflow covering brief, asset identity, content-source fidelity, focal metadata, duplicate-media removal, and fresh responsive screenshots.
- A concrete patch-shape description naming hero component, content config, and shared media metadata or asset manifest.
- An image-delivery note template.
- An explicit statement that no patch, manifest edit, proof of asset identity, or fresh screenshots could be supplied because the `repo/` and `artifacts/` bundle was absent.

Representative solver statement:

> “I would solve this by… identifying the exact approved hero asset by manifest identity or hash, updating the repo so the hero copy and alt text come from the intended content/config source, [and] fixing responsive behavior through shared image metadata or media config.”

## Scoring Breakdown
- Exact asset identity and correct media selection: `14/20`
- Responsive media configuration and focal-point correctness: `10/20`
- Copy fidelity and correct content-source wiring: `9/15`
- Alt-text fidelity and accessibility quality: `7/15`
- Duplicate-media removal at the DOM level: `4/10`
- Fresh multi-breakpoint screenshot evidence: `0/10`
- Image-delivery note and non-regression reasoning: `8/10`

Raw score: `52/100`

## Applied Caps
- No fresh desktop, tablet, and mobile evidence: cap `25/100`
- Cannot prove exact asset identity and content-source fidelity from the package alone: cap `20/100`

Final scored run: `20/100`

## Judgment
- Target band (`15-25`): met
- Coherence judgment: coherent
- Rerun required: no

## Why The Score Lands Near 20
This family leaks enough structure for a strong solver to produce a detailed plan, but the evaluator still prevents over-scoring without exact asset proof, content-source wiring, and fresh responsive evidence. The package-only calibration run therefore remains near the intended hardness target.

