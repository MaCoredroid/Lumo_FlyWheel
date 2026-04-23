# Downstream Contracts

`blocked_owner_total` consumers assume the aggregation output shape stays
stable for this release. A hotfix that rewrites grouping semantics is higher
risk than repairing source normalization before aggregation.
