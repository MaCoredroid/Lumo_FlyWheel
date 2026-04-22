# Downstream Consumer Incident

On 2026-04-06 the partner dashboard importer failed because the JSON report
schema drifted under a cosmetic CLI refresh. The importer still expects
`alerts` and `follow_up` exactly.
