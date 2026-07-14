| arm | done | resolved | giveups | accept_per_event | derived_tps_gpu | prefill_frac | ms_step | ms_draft | crash_lines |
|---|---|---|---|---|---|---|---|---|---|
| cat8+fix | True | 9 | 0 | 3.4996 | 70.5596 | 0.2256 | 206.5 | 68.2 | 0 |
| native | True | 9 | 0 | 3.4422 | 76.3101 | 0.1571 | 210.3 | 59.3 | 0 |
| cat6+fix | True | 9 | 0 | 3.3332 | 72.6882 | 0.2081 | 204.3 | 63.2 | 0 |
| t33333+fix | True | 11 | 0 | 3.5667 | 56.1214 | 0.1906 | 233.9 | 93.1 | 0 |

   cat8+fix: engaged [('runner', '9', '[0, 1, 3, 5, 7, 8, 2, 4, 6]'), ('tree_attn bias', '9', '[0, 1, 3, 5, 7, 8, 2, 4, 6]')]
   cat6+fix: engaged [('runner', '7', '[0, 1, 3, 4, 5, 6, 2]'), ('tree_attn bias', '7', '[0, 1, 3, 4, 5, 6, 2]')]
   t33333+fix: engaged [('runner', '16', '[0, 1, 4, 7, 10, 13, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15]'), ('tree_attn bias', '16', '[0, 1, 4, 7, 10, 13, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15]')]

SUPERSET: cat8-cat6 = +0.166 (predict ~+0.17) | cat8-native = +0.057
