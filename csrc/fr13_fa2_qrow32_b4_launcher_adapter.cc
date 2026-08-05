struct CUstream_st;

namespace flash {

struct Flash_fwd_params;

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32(
    Flash_fwd_params &params,
    CUstream_st *stream);

__attribute__((visibility("hidden")))
void fr13_run_mha_fwd_fixed32_qrow32_gqa_pair(
    Flash_fwd_params &params,
    CUstream_st *stream) {
  fr13_run_mha_fwd_fixed32_qrow32(params, stream);
}

}  // namespace flash
