# Provider rollover

Select `responses_proxy` as the maintained default provider.
It must point at the proxy-backed Responses path `http://127.0.0.1:11434/v1/responses`.
Do not select the legacy direct endpoint.
This repair applies to the `maintenance-responses` profile called out in release context.
Keep the docs grounded in the proxy-backed Responses path.
A rollback already occurred because the earlier hotfix left `store = true` unchecked in follow-up retrieval.
Call out the rollback explicitly so the same failure mode does not recur.
