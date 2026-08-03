# Fixed32 B4 N5120 hybrid CUTLASS host audit

This reduced artifact records a host-only CUDA 13.0 SM121a compile, link,
import, resource, SASS, and exact-geometry audit of a default-off B4 target
GEMM candidate. The selector routes the two exact `N=5120` projections to the
existing `128x128x128` cooperative identity Stage2 kernel and retains the
qualified `64x128x128` two-M ping-pong identity Stage2 kernel for the other
three exact projections.

## Result

The candidate is retained for real-task byte qualification. Both routed
kernels compile for FP16 and BF16 at 168 registers with zero stack, local
memory, `LDL`, `STL`, or `CALL`. The two-M kernel emits 1,032 SASS instructions
per dtype; the cooperative kernel emits 1,352. Both retain 128 QMMA, 128 FFMA,
48 LDSM, and 16 STSM instructions.

At the B4 48-SM geometry, each `N=5120` shape changes from 80 logical two-M
work tiles over 40 persistent workers to 40 cooperative work tiles over the
same 40 workers. This removes one complete scheduler work iteration per
worker. Across one instance of each exact projection shape, logical work tiles
fall from 1,184 to 1,104, a reduction of 80 or 6.7568%. Arithmetic work and
ordered full-K accumulation are unchanged.

The linked FP16 and BF16 callers for both routed kernels each query
`cudaDevAttrMultiProcessorCount`, store the result in scheduler parameters,
and call the corresponding scheduler `get_grid_shape` implementation.
The pinned CUTLASS exact-specialization trace independently confirms the full
dataflow: each specialization reads `Arguments::hw_info`, queries the device
when `sm_count <= 0`, builds a populated local `KernelHardwareInfo`, stores it
in both scheduler and GEMM Params, then forwards `Params::hw_info` to
`TileScheduler::get_grid_shape`. See `scheduler_hw_info_dataflow.tsv`.

The byte diagnostic remains stock-serving. Production dispatch directly calls
one of the two candidate kernels for an admitted exact shape; non-admitted
dispatch remains stock. The selector is default-off and is not production
authorized by this host audit.

No GPU kernel, Docker service, synthetic workload, SWE-Verified task, timing
run, or hardware-floor measurement was used. The object, cubin, linked binary,
generated dispatch, raw resource dump, raw disassembly, and build logs are not
published.
