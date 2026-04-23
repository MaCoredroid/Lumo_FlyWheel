# Auto-Research Iteration 001 of Round qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z

You are running ONE iteration of an auto-research round. Python is
running the round and will spawn your successor when you exit cleanly.

Round identity:
- round_id: qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z
- model_id: qwen3.5-27b
- family_id: proposal-ranking-manager-judgment
- active_layer: L1
- round_branch: autoresearch/qwen3.5-27b/proposal-ranking-manager-judgment/sprint-0/20260423T224422Z
- round_spec_ref: /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/round_spec.yaml

This iteration:
- iteration: 001
- iteration_dir: /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/candidates/001
- prior_results_ref: /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/results.tsv

Steps:
1. Read round_spec.yaml and results.tsv.
2. Write one candidate to /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/candidates/001/candidate.yaml.
3. Run:
   lumoserve auto-research measure --round-id qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z --candidate /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/candidates/001/candidate.yaml
4. Read /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/candidates/001/measurement_trace.json and then run:
   lumoserve auto-research commit-candidate --round-id qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z --iteration 001 --status <status> --notes "<one-line rationale>"
5. Exit 0.

Hard rules:
- Write only under /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/candidates/001.
- Do not modify round_spec.yaml, iteration_brief.md, results.tsv directly, src/, docs/, or benchmark_blueprints/.
- Do not call finalize-round.
- If a CLI call keeps failing, write /home/mark/shared/lumoFlyWheel/output/auto_research/qwen3.5-27b-proposal-ranking-manager-judgment-sprint-0-20260423T224422Z/candidates/001/BLOCKED.md and exit 2.


Runtime routing for this live round:
- Use the repo CLI explicitly: /home/mark/shared/lumoFlyWheel/.venv/bin/lumoserve --port 8100 --proxy-port 8101 ...
- Use absolute candidate paths under the iteration_dir above.
- Do not use synthetic fixtures or LUMO_AUTO_RESEARCH_ALLOW_NON_AGENT.
- Keep the response concise and terminal; the Python parent will decide whether to continue.
