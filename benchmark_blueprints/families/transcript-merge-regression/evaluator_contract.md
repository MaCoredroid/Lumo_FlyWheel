# Evaluator Contract

## Scope

- Family: `transcript-merge-regression`
- Task: reducer repair with incident-summary preservation

## Scoring breakdown

- `20`: visible tests pass
- `15`: same-name tool outputs survive by stable event identity
- `10`: interleaved fragments merge by `event_id`
- `10`: legitimate deferred post-completion tool output survives
- `10`: debug-only post-completion noise is ignored
- `10`: incident summary counts merged events, not rendered lines
- `5`: merge code explicitly keys by stable event identity
- `5`: merge logic distinguishes debug-only from legitimate deferred output
- `5`: incident note explains the safe boundary of the fix
- `10`: incident note variant grounding and shortcut rejection (`P_only`)

## Ceilings

- cap at `10` for note-only submissions
- cap at `20` for render-layer duplicate filtering
- cap at `20` for dropping all post-completion fragments
- cap at `25` if same-name tool outputs still collide
- cap at `40` if the summary still counts rendered lines
- cap at `60` if the incident note stays stale

## Hidden checks

- same-name tool outputs with distinct `event_id`s both survive merge
- interleaved fragments merge by `event_id` rather than tool name
- legitimate deferred post-completion tool output survives
- debug-only post-completion fragments do not render
- summary remains grounded in merged events
