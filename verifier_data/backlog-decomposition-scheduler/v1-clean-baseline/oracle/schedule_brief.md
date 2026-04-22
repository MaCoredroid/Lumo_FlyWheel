# Schedule Brief

- variant: `v1-clean-baseline`
- objective: Land the safe wave-1 cutover.

## Scheduled Slots

| slot | lane | item | summary |
|---:|---|---|---|
| 1 | core | `B1` | Place dependency-map in slot 1 because it respects the live dependency, capacity, and rollout constraints. |
| 2 | core | `B2` | Place customer-id-backfill in slot 2 because it respects the live dependency, capacity, and rollout constraints. |
| 2 | ops | `B6` | Place support-playbook in slot 2 because it respects the live dependency, capacity, and rollout constraints. |
| 3 | platform | `B4` | Place cutover-observability in slot 3 because it respects the live dependency, capacity, and rollout constraints. |
| 4 | core | `B3` | Place shadow-dry-run in slot 4 because it respects the live dependency, capacity, and rollout constraints. |
| 5 | release | `B5` | Place wave1-cutover in slot 5 because it respects the live dependency, capacity, and rollout constraints. |

## Scarce Role Plan

- role: `migration-sre`
- protected_items: B2, B4
- note: Keep migration-SRE work in different slots so Priya is not double-booked.

## Risk Gate

- risky_item_id: `B5`
- must_follow: B3, B4
- note: Wave-1 cutover waits for the dry-run and observability gates.

## Assumption Ledger

| status | topic | note |
|---|---|---|
| to_verify | Customer slice size | Need PM sign-off on the exact wave-1 batch size. |
| missing | Rollback staffing coverage | No document names the backup approver if Priya is unexpectedly unavailable. |
