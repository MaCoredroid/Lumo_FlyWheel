// Standalone harness for vllm-project/vllm PR #55122 PORT head 85f61e24b.
// Includes the PR's persistent_topk.cuh UNMODIFIED (byte-for-byte as fetched) and
// transcribes the host-side dispatch chain of launch_persistent_topk() from
// csrc/libtorch_stable/topk.cu line by line.
// Documented substitutions (see report):
//   S1  torch::stable::Tensor accessors -> raw device pointers + explicit
//       num_rows / stride / max_seq_len scalars (logits.size(0), logits.stride(0),
//       workspace.numel() become parameters).
//   S2  STD_TORCH_CHECK(...) -> non-throwing REJECT(...) recorder, so a host-side
//       rejection is distinguishable from a CUDA error. Only the
//       ctas_per_group > kDetMaxCtasPerGroup check is classified EXPECTED; every
//       other rejection is UNEXPECTED and fails the run.
//   S3  DeviceGuard + get_current_cuda_stream() -> device 0 and the default
//       stream; get_device_prop() -> cudaGetDeviceProperties(.,0).
//   S4  the two branches that are unreachable on this device (sampled_topk,
//       num_rows>64 && optin>=144KiB; FilteredTopK, num_rows>32 && optin>=128KiB;
//       and the FilteredTopK overflow fallback, optin>=128KiB) are not modelled;
//       they are recorded as UNEXPECTED rejections if ever selected.
//   S5  added routing diagnostics (Diag) and an explicit cudaDeviceSynchronize
//       after launch. No arithmetic is altered.
// The persistent (else) branch of launch_persistent_topk is BYTE-IDENTICAL between
// 7cfd04a3 and 85f61e24b (sha256 6970950770f87774a4f06fa948446992354ba0f9a4b6c10a340feb77d85015ea),
// so this transcription is unchanged from the 2026-09-11 run except for S4's new
// sampled_topk arm and the rejection classification in S2.
#include "persistent_topk.cuh"
#include "sampled_topk.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

namespace P = vllm::persistent;

#define CUDA_OK(expr)                                                      \
  do {                                                                     \
    cudaError_t _e = (expr);                                               \
    if (_e != cudaSuccess) {                                               \
      fprintf(stderr, "FATAL CUDA %s:%d: %s -> %s\n", __FILE__, __LINE__,  \
              #expr, cudaGetErrorString(_e));                              \
      exit(97);                                                            \
    }                                                                      \
  } while (0)

struct Diag {
  int num_sms = 0, max_smem_per_block = 0;
  int effective_max_smem = 0;
  unsigned vec_size = 0;
  size_t static_smem = 0;
  size_t available_for_ordered = 0;
  unsigned max_chunk_elements = 0;
  unsigned active_width = 0;
  unsigned ctas_per_group0 = 0;   // before any fallback edit
  unsigned chunk_size0 = 0;
  size_t smem_size_pre_det = 0;
  size_t det_want = 0, dyn_cap = 0;
  size_t smem_size = 0;
  int occupancy = 0;
  int needs_cooperative = 0;
  unsigned hw_resident_cap = 0, max_resident_ctas = 0;
  unsigned num_groups = 0, total_ctas = 0;
  unsigned ctas_per_group = 0, chunk_size = 0;
  unsigned force_single_cta = 0;
  int took_filtered = 0;
  int took_sampled = 0;
  int status = 0;   // 0 ok, 1 = pre-launch rejection (host check), 2 = cuda error
  int expected_rejection = 0;  // 1 only for the exact ctas_per_group > 64 condition
  char msg[512];
};

// Any host-side rejection other than the exact >64-CTA condition is UNEXPECTED
// and must fail the run (Codex protocol correction).
#define REJECT(D, ...)                          \
  do {                                          \
    (D).status = 1;                             \
    (D).expected_rejection = 0;                 \
    snprintf((D).msg, sizeof((D).msg), __VA_ARGS__); \
    return 1;                                   \
  } while (0)

#define REJECT_EXPECTED(D, ...)                 \
  do {                                          \
    (D).status = 1;                             \
    (D).expected_rejection = 1;                 \
    snprintf((D).msg, sizeof((D).msg), __VA_ARGS__); \
    return 1;                                   \
  } while (0)

// ---------------------------------------------------------------------------
// Verbatim transcription of the non-FilteredTopK branch of launch_persistent_topk
// ---------------------------------------------------------------------------
template <int TopK>
int launch_shim(const float* d_logits, const int32_t* d_lengths,
                int32_t* d_output, uint8_t* d_workspace, int64_t ws_bytes,
                int64_t num_rows, int64_t stride, int64_t max_seq_len,
                bool dry_run, Diag& D) {
  cudaStream_t stream = 0;
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;
  const int max_smem_per_block = prop.sharedMemPerBlockOptin;
  D.num_sms = num_sms;
  D.max_smem_per_block = max_smem_per_block;
  D.msg[0] = 0;

  // NEW IN THE PORT (85f61e24b): sampled_topk primary arm, ahead of FilteredTopK.
  // Transcribed condition; unreachable on this device (optin 101376 < 147456).
  if (num_rows > 64 &&
      max_seq_len >= vllm::sampled_topk::kMinSampledLength<TopK> &&
      max_smem_per_block >= 144 * 1024) {
    D.took_sampled = 1;
    REJECT(D, "sampled_topk primary path (not modelled here)");
  } else if (num_rows > 32 && max_smem_per_block >= 128 * 1024) {
    D.took_filtered = 1;
    REJECT(D, "FilteredTopKRaggedTransform primary path (not modelled here)");
  }

  int effective_max_smem;
  if (num_rows <= 4) {
    effective_max_smem = std::min(max_smem_per_block, (int)P::kSmemMedium);
  } else if (num_rows <= 8) {
    constexpr int kSmemCapMedium = 48 * 1024;
    effective_max_smem = std::min(max_smem_per_block, kSmemCapMedium);
  } else {
    effective_max_smem = max_smem_per_block;
  }
  D.effective_max_smem = effective_max_smem;

  uint32_t vec_size = 1;
  if (stride % 4 == 0) vec_size = 4;
  else if (stride % 2 == 0) vec_size = 2;
  D.vec_size = vec_size;

  cudaFuncAttributes chunk_fa{};
  cudaError_t chunk_fa_err =
      (vec_size == 4) ? cudaFuncGetAttributes(&chunk_fa, P::persistent_topk_kernel<TopK, 4>)
      : (vec_size == 2) ? cudaFuncGetAttributes(&chunk_fa, P::persistent_topk_kernel<TopK, 2>)
                        : cudaFuncGetAttributes(&chunk_fa, P::persistent_topk_kernel<TopK, 1>);
  if (chunk_fa_err != cudaSuccess)
    REJECT(D, "cudaFuncGetAttributes failed: %s", cudaGetErrorString(chunk_fa_err));
  const size_t static_smem = chunk_fa.sharedSizeBytes;
  D.static_smem = static_smem;
  size_t available_for_ordered =
      (size_t)effective_max_smem - P::kFixedSmemLarge - static_smem;
  D.available_for_ordered = available_for_ordered;
  uint32_t max_chunk_elements = (uint32_t)(available_for_ordered / sizeof(uint32_t));
  max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
  uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
  if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;
  D.max_chunk_elements = max_chunk_elements;

  uint32_t force_single_cta = 0u;
  const uint32_t active_width =
      std::min((uint32_t)stride, (uint32_t)std::max<int64_t>(max_seq_len, 0));
  D.active_width = active_width;
  uint32_t ctas_per_group = (active_width <= P::RADIX_THRESHOLD)
                                ? 1u
                                : (active_width + max_chunk_elements - 1) / max_chunk_elements;
  if (ctas_per_group == 0) ctas_per_group = 1;
  uint32_t chunk_size = (active_width + ctas_per_group - 1) / ctas_per_group;
  if (chunk_size == 0) chunk_size = max_chunk_elements;
  chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
  if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;
  D.ctas_per_group0 = ctas_per_group;
  D.chunk_size0 = chunk_size;

  size_t smem_size = P::kFixedSmemLarge + (size_t)chunk_size * sizeof(uint32_t);
  if (ctas_per_group > P::kDetMaxCtasPerGroup)
    REJECT_EXPECTED(D, "persistent_topk: ctas_per_group %u exceeds %u", ctas_per_group,
           P::kDetMaxCtasPerGroup);
  if (!((uint32_t)max_seq_len <= P::RADIX_THRESHOLD || chunk_size >= (uint32_t)TopK))
    REJECT(D, "persistent_topk: chunk_size %u smaller than TopK %d on the cooperative path",
           chunk_size, TopK);
  if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;
  D.smem_size_pre_det = smem_size;
  {
    const uint32_t det_rows = std::min<uint32_t>((uint32_t)max_seq_len, P::RADIX_THRESHOLD);
    const size_t det_want = P::det_select_row_bytes<TopK, P::kThreadsPerBlock>(det_rows);
    cudaFuncAttributes fa{};
    cudaError_t fa_err =
        (vec_size == 4) ? cudaFuncGetAttributes(&fa, P::persistent_topk_kernel<TopK, 4>)
        : (vec_size == 2) ? cudaFuncGetAttributes(&fa, P::persistent_topk_kernel<TopK, 2>)
                          : cudaFuncGetAttributes(&fa, P::persistent_topk_kernel<TopK, 1>);
    if (fa_err != cudaSuccess)
      REJECT(D, "cudaFuncGetAttributes failed: %s", cudaGetErrorString(fa_err));
    const size_t dyn_cap = (size_t)max_smem_per_block - fa.sharedSizeBytes;
    D.det_want = det_want;
    D.dyn_cap = dyn_cap;
    if (det_want > smem_size) smem_size = std::min(det_want, dyn_cap);
    if (smem_size > dyn_cap)
      REJECT(D, "persistent_topk: dynamic smem %zu exceeds %zu", smem_size, dyn_cap);
  }
  D.smem_size = smem_size;

  int occupancy = 1;
  cudaError_t occ_err = cudaSuccess;
  if (vec_size == 4)
    occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, P::persistent_topk_kernel<TopK, 4>, P::kThreadsPerBlock, smem_size);
  else if (vec_size == 2)
    occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, P::persistent_topk_kernel<TopK, 2>, P::kThreadsPerBlock, smem_size);
  else
    occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, P::persistent_topk_kernel<TopK, 1>, P::kThreadsPerBlock, smem_size);
  if (occ_err != cudaSuccess)
    REJECT(D, "occupancy query failed: %s", cudaGetErrorString(occ_err));
  if (occupancy < 1) occupancy = 1;
  D.occupancy = occupancy;

  const bool needs_cooperative = (uint32_t)max_seq_len > P::RADIX_THRESHOLD;
  D.needs_cooperative = needs_cooperative ? 1 : 0;

  const uint32_t hw_resident_cap = (uint32_t)num_sms * (uint32_t)occupancy;
  uint32_t max_resident_ctas = hw_resident_cap;
  if (needs_cooperative) {
    uint32_t headroom = (occupancy > 1) ? (uint32_t)num_sms : 1u;
    if (max_resident_ctas >= headroom + ctas_per_group) max_resident_ctas -= headroom;
  }
  uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group, (uint32_t)num_rows);
  if (num_groups == 0) num_groups = 1;
  uint32_t total_ctas = num_groups * ctas_per_group;
  D.hw_resident_cap = hw_resident_cap;
  D.max_resident_ctas = max_resident_ctas;

  if (needs_cooperative && total_ctas > hw_resident_cap) {
    if (max_smem_per_block < 128 * 1024) {
      force_single_cta = 1u;
      ctas_per_group = 1u;
      chunk_size = max_chunk_elements;
      num_groups = std::min(max_resident_ctas, (uint32_t)num_rows);
      if (num_groups == 0) num_groups = 1;
      total_ctas = num_groups;
    } else {
      D.took_filtered = 1;
      REJECT(D, "FilteredTopK fallback (not modelled here)");
    }
  }
  D.num_groups = num_groups;
  D.total_ctas = total_ctas;
  D.ctas_per_group = ctas_per_group;
  D.chunk_size = chunk_size;
  D.force_single_cta = force_single_cta;

  if (dry_run) return 0;

  size_t state_bytes = (size_t)num_groups * sizeof(P::RadixRowState);
  if ((int64_t)state_bytes > ws_bytes)
    REJECT(D, "workspace too small, need %zu bytes, have %lld", state_bytes,
           (long long)ws_bytes);
  cudaError_t mz = cudaMemsetAsync(d_workspace, 0, state_bytes, stream);
  if (mz != cudaSuccess) REJECT(D, "row_states memset failed: %s", cudaGetErrorString(mz));

  P::PersistentTopKParams params;
  params.input = d_logits;
  params.output = d_output;
  params.lengths = d_lengths;
  params.num_rows = (uint32_t)num_rows;
  params.stride = (uint32_t)stride;
  params.top_k = (uint32_t)TopK;
  params.chunk_size = chunk_size;
  params.row_states = reinterpret_cast<P::RadixRowState*>(d_workspace);
  params.ctas_per_group = ctas_per_group;
  params.max_seq_len = (uint32_t)max_seq_len;
  params.det_smem_bytes = (uint32_t)smem_size;
  params.force_single_cta = force_single_cta;

#define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                     \
  do {                                                                      \
    auto kernel = &P::persistent_topk_kernel<TOPK_VAL, VS>;                 \
    cudaError_t err = cudaFuncSetAttribute(                                 \
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);    \
    if (err != cudaSuccess)                                                 \
      REJECT(D, "Failed to set smem: %s", cudaGetErrorString(err));         \
    kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(params); \
  } while (0)

  if (vec_size == 4) { LAUNCH_PERSISTENT(TopK, 4); }
  else if (vec_size == 2) { LAUNCH_PERSISTENT(TopK, 2); }
  else { LAUNCH_PERSISTENT(TopK, 1); }
#undef LAUNCH_PERSISTENT

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    D.status = 2;
    snprintf(D.msg, sizeof(D.msg), "persistent_topk launch failed: %s",
             cudaGetErrorString(err));
    return 2;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    D.status = 2;
    snprintf(D.msg, sizeof(D.msg), "persistent_topk sync failed: %s",
             cudaGetErrorString(err));
    return 2;
  }
  return 0;
}

static int dispatch_launch(int topk, const float* a, const int32_t* b, int32_t* c,
                           uint8_t* w, int64_t wb, int64_t rows, int64_t stride,
                           int64_t msl, bool dry, Diag& D) {
  if (topk == 512) return launch_shim<512>(a, b, c, w, wb, rows, stride, msl, dry, D);
  if (topk == 1024) return launch_shim<1024>(a, b, c, w, wb, rows, stride, msl, dry, D);
  if (topk == 2048) return launch_shim<2048>(a, b, c, w, wb, rows, stride, msl, dry, D);
  fprintf(stderr, "unsupported k %d\n", topk);
  exit(96);
}

static void print_diag(const char* tag, const Diag& D) {
  printf("DIAG %s sms=%d optin=%d eff_smem=%d vec=%u static_smem=%zu avail=%zu "
         "max_chunk=%u active_w=%u ctas0=%u chunk0=%u smem_pre_det=%zu det_want=%zu "
         "dyn_cap=%zu smem=%zu occ=%d coop=%d hw_cap=%u max_res=%u groups=%u "
         "total_ctas=%u ctas_pg=%u chunk=%u FORCE_SINGLE_CTA=%u status=%d "
         "took_sampled=%d took_filtered=%d expected_rej=%d msg=\"%s\"\n",
         tag, D.num_sms, D.max_smem_per_block, D.effective_max_smem, D.vec_size,
         D.static_smem, D.available_for_ordered, D.max_chunk_elements, D.active_width,
         D.ctas_per_group0, D.chunk_size0, D.smem_size_pre_det, D.det_want, D.dyn_cap,
         D.smem_size, D.occupancy, D.needs_cooperative, D.hw_resident_cap,
         D.max_resident_ctas, D.num_groups, D.total_ctas, D.ctas_per_group,
         D.chunk_size, D.force_single_cta, D.status, D.took_sampled,
         D.took_filtered, D.expected_rejection, D.msg);
}

// ---------------------------------------------------------------------------
// input generation + exact stable reference
// ---------------------------------------------------------------------------
enum Pattern { PAT_RANDOM = 0, PAT_TIE = 1, PAT_EQUAL = 2 };

static void gen_row(std::vector<float>& row, int n, Pattern pat, uint64_t seed) {
  std::mt19937_64 rng(seed);
  row.assign(n, 0.0f);
  if (pat == PAT_RANDOM) {
    std::uniform_real_distribution<float> d(-100.0f, 100.0f);
    for (int i = 0; i < n; i++) row[i] = d(rng);
  } else if (pat == PAT_TIE) {
    // randint(0,5) -- the suite's tie-heavy structure: massive tie populations
    std::uniform_int_distribution<int> d(0, 4);
    for (int i = 0; i < n; i++) row[i] = (float)d(rng);
  } else {
    for (int i = 0; i < n; i++) row[i] = 1.5f;
  }
}

// value-descending, index-ascending stable; output = selected indices sorted ascending
static void reference_topk(const float* row, int n, int k, std::vector<int32_t>& out) {
  std::vector<int32_t> idx(n);
  for (int i = 0; i < n; i++) idx[i] = i;
  int kk = std::min(k, n);
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int32_t a, int32_t b) { return row[a] > row[b]; });
  out.assign(idx.begin(), idx.begin() + kk);
  std::sort(out.begin(), out.end());
  while ((int)out.size() < k) out.push_back(-1);
}

static const int32_t POISON = -424242;

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage:\n"
            "  %s route <rows> <k> <n1,n2,...>\n"
            "  %s run <rows> <k> <n> <pattern:random|tie|equal> <repeats> <seed>\n",
            argv[0], argv[0]);
    return 2;
  }
  CUDA_OK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  printf("DEVICE name=%s cc=%d.%d sms=%d sharedPerBlockOptin=%zu "
         "sharedPerBlock=%zu sharedPerMultiprocessor=%zu\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount,
         prop.sharedMemPerBlockOptin, prop.sharedMemPerBlock,
         prop.sharedMemPerMultiprocessor);
  printf("CONST kThreadsPerBlock=%d RADIX_THRESHOLD=%u kSmemMedium=%zu "
         "kFixedSmemLarge=%zu kDetMaxCtasPerGroup=%u det_fixed512=%zu "
         "det_fixed1024=%zu det_fixed2048=%zu\n",
         P::kThreadsPerBlock, P::RADIX_THRESHOLD, P::kSmemMedium, P::kFixedSmemLarge,
         P::kDetMaxCtasPerGroup,
         P::det_select_row_fixed_bytes<512, P::kThreadsPerBlock>(),
         P::det_select_row_fixed_bytes<1024, P::kThreadsPerBlock>(),
         P::det_select_row_fixed_bytes<2048, P::kThreadsPerBlock>());

  std::string mode = argv[1];

  if (mode == "route") {
    int rows = atoi(argv[2]);
    int k = atoi(argv[3]);
    std::string list = argv[4];
    size_t pos = 0;
    while (pos <= list.size()) {
      size_t c = list.find(',', pos);
      std::string tok = list.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
      if (!tok.empty()) {
        long n = atol(tok.c_str());
        Diag D{};
        dispatch_launch(k, nullptr, nullptr, nullptr, nullptr, 0, rows, n, n, true, D);
        char tag[128];
        snprintf(tag, sizeof(tag), "rows=%d k=%d n=%ld", rows, k, n);
        print_diag(tag, D);
      }
      if (c == std::string::npos) break;
      pos = c + 1;
    }
    return 0;
  }

  if (mode != "run") { fprintf(stderr, "bad mode\n"); return 2; }
  int rows = atoi(argv[2]);
  int k = atoi(argv[3]);
  long n = atol(argv[4]);
  std::string ps = argv[5];
  int repeats = atoi(argv[6]);
  uint64_t seed = strtoull(argv[7], nullptr, 10);
  Pattern pat = ps == "random" ? PAT_RANDOM : (ps == "tie" ? PAT_TIE : PAT_EQUAL);

  const int64_t stride = n;  // exact pitch, multiple of 4 assumed
  std::vector<float> host(rows * (size_t)n);
  for (int r = 0; r < rows; r++) {
    std::vector<float> row;
    gen_row(row, n, pat, seed * 1000003ull + r);
    memcpy(&host[(size_t)r * n], row.data(), sizeof(float) * n);
  }
  std::vector<int32_t> hlen(rows, (int32_t)n);

  float* d_in = nullptr; int32_t* d_len = nullptr; int32_t* d_out = nullptr;
  uint8_t* d_ws = nullptr;
  const int64_t ws_bytes = (int64_t)4096 * sizeof(P::RadixRowState);
  CUDA_OK(cudaMalloc(&d_in, sizeof(float) * host.size()));
  CUDA_OK(cudaMalloc(&d_len, sizeof(int32_t) * rows));
  CUDA_OK(cudaMalloc(&d_out, sizeof(int32_t) * (size_t)rows * k));
  CUDA_OK(cudaMalloc(&d_ws, ws_bytes));
  CUDA_OK(cudaMemcpy(d_in, host.data(), sizeof(float) * host.size(), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_len, hlen.data(), sizeof(int32_t) * rows, cudaMemcpyHostToDevice));

  // exact reference
  std::vector<std::vector<int32_t>> ref(rows);
  for (int r = 0; r < rows; r++) reference_topk(&host[(size_t)r * n], n, k, ref[r]);

  std::vector<int32_t> poison((size_t)rows * k, POISON);
  std::vector<int32_t> got((size_t)rows * k);
  std::vector<int32_t> first;
  int all_pass = 1;

  for (int rep = 0; rep < repeats; rep++) {
    CUDA_OK(cudaMemcpy(d_out, poison.data(), sizeof(int32_t) * poison.size(),
                       cudaMemcpyHostToDevice));
    Diag D{};
    cudaEvent_t e0, e1; CUDA_OK(cudaEventCreate(&e0)); CUDA_OK(cudaEventCreate(&e1));
    CUDA_OK(cudaEventRecord(e0));
    int st = dispatch_launch(k, d_in, d_len, d_out, d_ws, ws_bytes, rows, stride, n,
                             false, D);
    CUDA_OK(cudaEventRecord(e1)); CUDA_OK(cudaEventSynchronize(e1));
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    char tag[160];
    snprintf(tag, sizeof(tag), "rows=%d k=%d n=%ld pat=%s rep=%d", rows, k, n, ps.c_str(), rep);
    if (rep == 0) print_diag(tag, D);
    if (st != 0) {
      const char* oc = (D.status == 2) ? "CUDA_ERROR"
                       : D.expected_rejection ? "REJECTED_PRELAUNCH_EXPECTED"
                                              : "REJECTED_PRELAUNCH_UNEXPECTED";
      printf("RESULT %s rep=%d outcome=%s detail=\"%s\"\n", tag, rep, oc, D.msg);
      if (D.status == 2) { printf("STOP device fault\n"); return 3; }
      if (!D.expected_rejection) { printf("STOP unexpected host rejection\n"); return 4; }
      all_pass = 1;  // the expected >64-CTA rejection is the intended outcome
      break;  // deterministic host rejection: no point repeating
    }
    CUDA_OK(cudaMemcpy(got.data(), d_out, sizeof(int32_t) * got.size(),
                       cudaMemcpyDeviceToHost));
    int poisoned = 0, exact = 1, repro = 1;
    for (size_t i = 0; i < got.size(); i++) if (got[i] == POISON) poisoned++;
    for (int r = 0; r < rows; r++)
      for (int j = 0; j < k; j++)
        if (got[(size_t)r * k + j] != ref[r][j]) { exact = 0; }
    if (rep == 0) first = got;
    else for (size_t i = 0; i < got.size(); i++) if (got[i] != first[i]) repro = 0;
    // first mismatch detail
    char detail[256]; detail[0] = 0;
    if (!exact) {
      for (int r = 0; r < rows && !detail[0]; r++)
        for (int j = 0; j < k; j++)
          if (got[(size_t)r * k + j] != ref[r][j]) {
            snprintf(detail, sizeof(detail), "row=%d pos=%d got=%d ref=%d", r, j,
                     got[(size_t)r * k + j], ref[r][j]);
            break;
          }
    }
    int pass = exact && repro && poisoned == 0;
    if (!pass) all_pass = 0;
    printf("RESULT %s rep=%d outcome=%s exact=%d repro=%d poison_left=%d ms=%.3f "
           "force_single_cta=%u detail=\"%s\"\n",
           tag, rep, pass ? "PASS" : "FAIL", exact, repro, poisoned, ms,
           D.force_single_cta, detail);
    fflush(stdout);
  }
  printf("SUMMARY rows=%d k=%d n=%ld pat=%s all_pass=%d\n", rows, k, n, ps.c_str(), all_pass);
  cudaFree(d_in); cudaFree(d_len); cudaFree(d_out); cudaFree(d_ws);
  return all_pass ? 0 : 1;
}
