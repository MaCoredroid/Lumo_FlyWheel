-- FR13 fixed32 B1 attack-ladder analysis: queries used.
-- Run against the sqlite export of
--   logs/fr13_fixed32_b1_real_swe.nsys-rep
-- produced by:
--   /opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys export --type sqlite \
--     --output fr13_fixed32_b1_real_swe.sqlite fr13_fixed32_b1_real_swe.nsys-rep
-- Open read-only:  sqlite3 "file:fr13_fixed32_b1_real_swe.sqlite?mode=ro"
--
-- Constants used below: 1146 complete fr13.fixed32.step instances,
-- 1147 sfwd/graph-812 replays. Divide capture totals by these for ms/step.
-- StringIds referenced: 155 = target GEMM (demangled), 270 = FA2 qrow16,
-- 216 = _tree_gdn_path_kernel, 157 = reshape_and_cache_flash_kernel.
-- Resolve them for a fresh export with section A.

-- ===========================================================================
-- A. Kernel identity
-- ===========================================================================
SELECT id, substr(value,1,110) FROM StringIds
WHERE value LIKE '%cutlass_3x_gemm_fp8_blockwise%'
   OR value LIKE '%flash_fwd_splitkv%'
   OR value LIKE '%_tree_gdn_path_kernel%'
   OR value LIKE '%reshape_and_cache%';

-- ===========================================================================
-- B. CUDA-graph inventory  (proves SFWD is one graph, and finds eager phases)
-- ===========================================================================
SELECT launchType, (graphId IS NOT NULL) has_graph, COUNT(*) n
FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1,2;

SELECT graphId,
       COUNT(*) nk,
       COUNT(DISTINCT correlationId) replays,
       COUNT(*)/COUNT(DISTINCT correlationId) nodes_per_replay,
       COUNT(DISTINCT demangledName) distinct_kernels,
       ROUND(SUM(end-start)/1e6,1) gpu_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId IS NOT NULL
GROUP BY 1 ORDER BY nk DESC;

-- single-stream check
SELECT streamId, COUNT(*) n, ROUND(SUM(end-start)/1e6,1) ms
FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1 ORDER BY n DESC;

-- ===========================================================================
-- C. Target GEMM: launch geometry and occupancy inputs
-- ===========================================================================
SELECT gridX,gridY,gridZ,blockX,blockY,blockZ,
       registersPerThread reg, staticSharedMemory ssm, dynamicSharedMemory dsm,
       clusterX,clusterY,clusterZ,
       COUNT(*) n, ROUND(AVG(end-start),1) avg_ns, ROUND(SUM(end-start)/1e6,3) tot_ms
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=155
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12 ORDER BY n DESC;

-- ===========================================================================
-- D. Target GEMM: per-shape duration distribution inside the SFWD graph.
--    gridY == CTA count in decode (gridX=gridZ=1).
--    gridY=40 is trimodal; the duration split at 160 us / 280 us separates
--    attn_o (16/step), gdn_o (48/step), mlp_down (64/step).
-- ===========================================================================
WITH cls AS (
  SELECT (end-start) dur,
    CASE WHEN gridY=272 THEN 'mlp_gate_up'
         WHEN gridY=128 THEN 'gdn_in_proj'
         WHEN gridY=112 THEN 'attn_qkv_gate'
         WHEN gridY=40 AND (end-start)>=280000 THEN 'mlp_down'
         WHEN gridY=40 AND (end-start)>=160000 THEN 'gdn_o_proj'
         ELSE 'attn_o_proj' END c
  FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=155 AND graphId=812),
r AS (SELECT c, dur, ROW_NUMBER() OVER (PARTITION BY c ORDER BY dur) rn,
             COUNT(*) OVER (PARTITION BY c) n FROM cls)
SELECT c, n, ROUND(n/1147.0,1) per_step,
  ROUND(MAX(CASE WHEN rn=CAST(n*0.05 AS INT)+1 THEN dur END)/1e3,1) p5_us,
  ROUND(MAX(CASE WHEN rn=CAST(n*0.50 AS INT)+1 THEN dur END)/1e3,1) p50_us,
  ROUND(AVG(dur)/1e3,1) mean_us,
  ROUND(SUM(dur)/1147.0/1e6,3) ms_step,
  ROUND(MAX(CASE WHEN rn=CAST(n*0.05 AS INT)+1 THEN dur END)*n/1147.0/1e6,3) ms_step_at_p5
FROM r GROUP BY c ORDER BY ms_step DESC;

-- variance decomposition: per-instance sigma vs per-step-mean sigma
SELECT ROUND(SQRT(AVG(1.0*(end-start)*(end-start))
                 -AVG(1.0*(end-start))*AVG(1.0*(end-start)))/1e3,2) sd_instance_us
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=155 AND graphId=812 AND gridY=272;

WITH pr AS (SELECT correlationId cid, AVG(end-start) m
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE demangledName=155 AND graphId=812 AND gridY=272 GROUP BY 1)
SELECT ROUND(MIN(m)/1e3,1) min_us, ROUND(AVG(m)/1e3,1) mean_us, ROUND(MAX(m)/1e3,1) max_us,
       ROUND(SQRT(AVG(m*m)-AVG(m)*AVG(m))/1e3,2) sd_of_step_means_us FROM pr;

-- no drift with position inside the replay
WITH pos AS (SELECT (end-start) dur, gridY gy,
       ROW_NUMBER() OVER (PARTITION BY correlationId ORDER BY start) p
       FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=812)
SELECT (p/200)*200 pos_bucket, COUNT(*) n, ROUND(AVG(dur)/1e3,1) avg_us
FROM pos WHERE gy=272 GROUP BY 1 ORDER BY 1;

-- ===========================================================================
-- E. Inter-kernel gaps.  Single stream, so gap = start - previous end.
-- ===========================================================================
WITH s AS (SELECT start, end, demangledName dn, correlationId cid,
            LAG(end) OVER (ORDER BY start) prev_end,
            ROW_NUMBER() OVER (PARTITION BY correlationId ORDER BY start) pos
           FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=812)
SELECT 'target GEMM' lbl, COUNT(*) n,
       ROUND(SUM(start-prev_end)/1147.0/1e3,2) us_per_step,
       ROUND(AVG(start-prev_end),1) avg_ns, MAX(start-prev_end) max_ns
FROM s WHERE dn=155 AND pos>1
UNION ALL
SELECT 'all sfwd nodes', COUNT(*),
       ROUND(SUM(start-prev_end)/1147.0/1e3,2),
       ROUND(AVG(start-prev_end),1), MAX(start-prev_end)
FROM s WHERE pos>1;

-- whole-replay span vs busy (the 0.404 ms/step SFWD gap total)
WITH g AS (SELECT correlationId cid, MIN(start) s, MAX(end) e, SUM(end-start) busy,
                  COUNT(*) n FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=812 GROUP BY 1)
SELECT COUNT(*) replays, ROUND(AVG(e-s)/1e6,4) span_ms, ROUND(AVG(busy)/1e6,4) busy_ms,
       ROUND(AVG(e-s-busy)/1e6,4) gap_ms, ROUND(AVG(n),1) nodes FROM g;

-- ===========================================================================
-- F. GDN level structure  (gridZ = n_paths: 1 at level 0, 11 at level 1)
-- ===========================================================================
SELECT gridX,gridY,gridZ,blockX,registersPerThread reg,dynamicSharedMemory dsm,
       COUNT(*) n, ROUND(COUNT(*)/1147.0,1) per_step,
       ROUND(MIN(end-start)/1e3,2) min_us, ROUND(AVG(end-start)/1e3,2) avg_us,
       ROUND(SUM(end-start)/1147.0/1e6,4) ms_step
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=216 AND graphId=812
GROUP BY 1,2,3,4,5,6;

-- level0 -> level1 handoff gap (back-to-back: ~0)
WITH s AS (SELECT start, gridZ gz, demangledName dn,
            LAG(end) OVER (ORDER BY start) pe,
            LAG(demangledName) OVER (ORDER BY start) pdn
           FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=812)
SELECT ROUND(AVG(start-pe),1) avg_gap_ns, COUNT(*) n
FROM s WHERE dn=216 AND gz=11 AND pdn=216;

-- ===========================================================================
-- G. FA2 qrow16
-- ===========================================================================
SELECT gridX,gridY,gridZ,blockX,registersPerThread reg,dynamicSharedMemory dsm,
       COUNT(*) n, ROUND(COUNT(*)/1147.0,2) per_step,
       ROUND(MIN(end-start)/1e3,1) min_us, ROUND(AVG(end-start)/1e3,1) avg_us,
       ROUND(MAX(end-start)/1e3,1) max_us, ROUND(SUM(end-start)/1147.0/1e6,4) ms_step
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=270 AND graphId=812
GROUP BY 1,2,3,4,5,6;

WITH s AS (SELECT start, demangledName dn, LAG(end) OVER (ORDER BY start) pe
           FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphId=812)
SELECT ROUND(AVG(start-pe),1) avg_gap_ns, MAX(start-pe) max_gap_ns, COUNT(*) n
FROM s WHERE dn=270;

-- context ramp: per-step mean FA2 duration in launch order.
-- OLS over these 1147 points gives intercept 1090.8 us, slope 0.4260 us/step.
SELECT ROW_NUMBER() OVER (ORDER BY MIN(start)) step,
       ROUND(AVG(end-start)/1e3,1) fa2_us
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=270 AND graphId=812
GROUP BY correlationId ORDER BY MIN(start);

-- KV cache dtype + tokens written per step (grid.x = 32 tree tokens)
SELECT graphId, gridX, gridY, gridZ, blockX, COUNT(*) n,
       ROUND(COUNT(*)/1147.0,2) per_step, ROUND(AVG(end-start),0) avg_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE demangledName=157 GROUP BY 1,2,3,4,5 ORDER BY n DESC;

-- ===========================================================================
-- H. Host side.  Phase/idle attribution needs the NVTX -> GPU projection:
--    NVTX host range [a,b)  ->  CUPTI_ACTIVITY_KIND_RUNTIME rows with
--    start in [a,b)  ->  their correlationId  ->  CUPTI_ACTIVITY_KIND_KERNEL.
--    (Graph replays correlate one cudaGraphLaunch to all of the graph's nodes,
--     which is why SFWD projects to 1894 GPU ops from a single runtime call.)
--    The reference implementation is a linear merge over both streams sorted
--    by host start; SQL sketch below is the same relation.
-- ===========================================================================
SELECT COALESCE(si.value, n.text) nm, COUNT(*) n
FROM NVTX_EVENTS n LEFT JOIN StringIds si ON si.id=n.textId
WHERE COALESCE(si.value,n.text) LIKE 'fr13.fixed32.%' GROUP BY 1;

-- per-phase span / busy / idle  (one phase range at a time; :a,:b from above)
--   SELECT MIN(k.start), MAX(k.end), SUM(k.end-k.start), COUNT(*)
--   FROM CUPTI_ACTIVITY_KIND_KERNEL k
--   JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
--   WHERE r.start >= :a AND r.start < :b;
-- busy must be computed as the union of [start,end) intervals, not the plain
-- sum, although on this single-stream trace the two agree.

-- host API census
SELECT si.value api, COUNT(*) n, ROUND(COUNT(*)/1146.0,1) per_step,
       ROUND(SUM(r.end-r.start)/1146.0/1e6,3) cpu_ms_step,
       ROUND(AVG(r.end-r.start)/1e3,2) mean_us
FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds si ON si.id=r.nameId
GROUP BY 1 ORDER BY cpu_ms_step DESC;

-- thread census (one CUDA thread carries everything)
SELECT globalTid, COUNT(*) n, ROUND(SUM(end-start)/1e6,0) cpu_ms
FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY 1 ORDER BY n DESC;

-- memcpy census
SELECT e.label kind, COUNT(*) n, ROUND(COUNT(*)/1146.0,1) per_step,
       SUM(m.bytes) tot_bytes, ROUND(AVG(m.bytes),0) mean_bytes,
       ROUND(SUM(m.end-m.start)/1146.0/1e6,3) gpu_ms_step
FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
GROUP BY 1 ORDER BY n DESC;

-- profiler overhead: which thread carries it
SELECT globalTid, COUNT(*) n, ROUND(AVG(end-start)/1e3,1) mean_us,
       ROUND(SUM(end-start)/1146.0/1e6,3) ms_step
FROM PROFILER_OVERHEAD GROUP BY 1;
