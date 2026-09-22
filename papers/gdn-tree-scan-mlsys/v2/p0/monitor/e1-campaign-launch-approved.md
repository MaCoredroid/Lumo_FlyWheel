# E1 launch approval after independent review

Decision time: 2026-09-22T09:59:38Z. This is the coordinator's internal evidence gate under the user's existing authorization.

The settled reply-25 campaign passed the independent review recorded in `e1-campaign-final-redteam.md` (SHA-256 `f10d099c370a78844aaa69f546da1ff24788c5896ac92a2a2aa91aecc22ffb4e`): 51 targeted CPU controls closed the terminal-seal, frozen-score, frozen-allocation, and qualification-metadata findings. The old failed-final-audit counterexample remains in that report.

The final change adds the actual dynamically loaded device multidraft kernel and fused-convolution module to the existing per-cell source identity guard. The coordinator inspected the narrow runner change and independently reran `test_e1_campaign_stub.sh`: all 20 checks passed, including refusal after mutating each of the two newly covered files and restoration of successful guard behavior. Evidence: `e1-reply26-parent-stub.json` and `.log`.

- Runner: `910e2dac38a8661361da7f011f8a6a48563b84f6799eb5b1257fc558fa765d72`.
- Device draft kernel: `648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9`.
- Fused convolution: `564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e`.
- Aggregate, copied by the parent at the first model load before timing results: `98beec9efac26f560783d9d77af3e2154edbd6b24ebadfe141b0cafce6e48a68` (`e1-parent-source-checkpoint1001.json`).

The campaign launched at 2026-09-22T10:00:28Z in `experiments/out-20260922T100028Z-e1-18cells/`, runner PID 3335600. Actual first-container configuration and source identity were independently checked. Native-5/B1 live preflight passed at 10:07:20Z and then entered fixed warmup/timing. No timing cell was yet complete at this note's creation.

Launch approval does not establish a measurement result. Native live preflight and final sealed-ledger audits, per-cell verification, source identity, support floors, and preservation of failed attempts remain mandatory. The campaign stops on INVALID; measured cells are not repeated for favorable outcomes. The initial 18-cell analysis and its prespecified precision limitations remain primary.
