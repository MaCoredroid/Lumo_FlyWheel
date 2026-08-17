// FR14 TreeAttention-v2 fallback F2 gate: can 3 CTAs on sm_121a share one
// staged tile through distributed shared memory?
//
// Attribute support (CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH == 1) is not the same
// claim as "a peer CTA can correctly read my smem", so this probe writes a
// known pattern in cluster rank 0 only and has ranks 1 and 2 read it back
// through cluster.map_shared_rank(). Cluster size is 3 because that is exactly
// the number of CTAs sharing one KV head at B1 today.
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define TILE_FLOATS 4096  // 16 KB, a stand-in for the staged K/V tile

__global__ void __cluster_dims__(3, 1, 1) dsmem_probe(int *out, float *sink) {
    extern __shared__ float smem[];
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned rank = cluster.block_rank();
    const int tid = threadIdx.x;

    // Only rank 0 stages the tile -- the whole point of the design.
    if (rank == 0) {
        for (int i = tid; i < TILE_FLOATS; i += blockDim.x) {
            smem[i] = (float)(i * 3 + 1);
        }
    }
    cluster.sync();

    // Every rank reads rank 0's smem through DSMEM and checks it.
    float *remote = cluster.map_shared_rank(smem, 0);
    int local_bad = 0;
    float acc = 0.0f;
    for (int i = tid; i < TILE_FLOATS; i += blockDim.x) {
        float v = remote[i];
        acc += v;
        if (v != (float)(i * 3 + 1)) local_bad++;
    }
    atomicAdd(&out[rank], local_bad);
    if (acc == 12345.678f) sink[0] = acc;  // keep the reads alive
    cluster.sync();
}

int main() {
    int *d_out;
    float *d_sink;
    cudaMalloc(&d_out, 3 * sizeof(int));
    cudaMalloc(&d_sink, sizeof(float));
    cudaMemset(d_out, 0, 3 * sizeof(int));

    size_t smem = TILE_FLOATS * sizeof(float);
    cudaFuncSetAttribute(dsmem_probe,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);

    // 12 CTAs = 4 clusters of 3: the B1 gqa_pair grid, regrouped.
    dsmem_probe<<<12, 256, smem>>>(d_out, d_sink);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync = cudaDeviceSynchronize();

    int h_out[3] = {-1, -1, -1};
    cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

    printf("{\"cluster_dims\": 3, \"grid\": 12, \"threads\": 256, "
           "\"tile_bytes\": %zu, \"launch\": \"%s\", \"sync\": \"%s\", "
           "\"mismatches_rank0\": %d, \"mismatches_rank1\": %d, "
           "\"mismatches_rank2\": %d, \"dsmem_functional\": %s}\n",
           smem, cudaGetErrorString(launch), cudaGetErrorString(sync),
           h_out[0], h_out[1], h_out[2],
           (launch == cudaSuccess && sync == cudaSuccess && h_out[0] == 0 &&
            h_out[1] == 0 && h_out[2] == 0)
               ? "true"
               : "false");
    return 0;
}
