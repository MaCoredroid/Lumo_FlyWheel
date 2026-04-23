# Duplicate Heartbeat Rollback

Incident INC-4412 rolled back a previous attempt that re-enabled
`automations/handoff-copy.toml` during an escalation follow-up and caused
duplicate pages. Recovery rule: keep the duplicate heartbeat retired and
document the rollback in the live runbook before restoring the follow-up skill.
