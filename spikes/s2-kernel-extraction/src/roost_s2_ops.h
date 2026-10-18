/* S2 spike (R-006): illustrative C entry points for the non-attention M0 ops (bf16 only).
 * Not contract C1. Every function returns a RoostS2Status; no allocation happens inside a call. */
#ifndef ROOST_S2_OPS_H_
#define ROOST_S2_OPS_H_
#include "roost_s2_attn.h"
#ifdef __cplusplus
extern "C" {
#endif
/* x, w, out: bf16. x is [rows, d] with row stride stride_x; d % 8 == 0. */
ROOST_S2_API int32_t roost_s2_rms_norm_flashinfer(const void* x, const void* w, void* out,
                                                  uint32_t rows, uint32_t d, uint32_t stride_x,
                                                  float eps, void* stream);
ROOST_S2_API int32_t roost_s2_rms_norm_vllm(const void* x, const void* w, void* out, uint32_t rows,
                                            uint32_t d, uint32_t stride_x, float eps, void* stream);
ROOST_S2_API int32_t roost_s2_fused_add_rms_norm(void* x, void* residual, const void* w,
                                                 uint32_t rows, uint32_t d, float eps, void* stream);
/* QkNorm: x is [tokens, heads, d] contiguous, one weight of length d shared by heads. */
ROOST_S2_API int32_t roost_s2_qk_rms_norm(const void* x, const void* w, void* out, uint32_t tokens,
                                          uint32_t heads, uint32_t d, float eps, void* stream);
/* in is [tokens, 2*d] (gate | up); out is [tokens, d]. */
ROOST_S2_API int32_t roost_s2_silu_and_mul(const void* in, void* out, uint32_t tokens, uint32_t d,
                                           void* stream);
/* In-place half-split (NeoX) rope on q [nnz, qh, hd] and k [nnz, kvh, hd]. */
ROOST_S2_API int32_t roost_s2_rope_neox(void* q, void* k, const int32_t* pos, uint32_t nnz,
                                        uint32_t qh, uint32_t kvh, uint32_t hd, float theta,
                                        void* stream);
/* Write nnz new tokens into the paged cache (NHD pools). batch_idx[i] and pos[i] locate token i. */
ROOST_S2_API int32_t roost_s2_append_kv(void* k_pool, void* v_pool, const int32_t* kv_indices,
                                        const int32_t* kv_indptr, const int32_t* last_page_len,
                                        const void* k, const void* v, const int32_t* batch_idx,
                                        const int32_t* pos, uint32_t batch, uint32_t nnz,
                                        uint32_t kvh, uint32_t hd, uint32_t page, void* stream);
#ifdef __cplusplus
}
#endif
#endif
