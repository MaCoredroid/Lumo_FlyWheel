# Incident Ack Policy

The incident queue highlights unacked pages that have crossed the ack SLA:

- `sev1`: 15 minutes
- `sev2`: 30 minutes
- `sev3`: 60 minutes

When an incident reaches the SLA threshold, it should already count as a
breach for handoff and escalation reporting.
