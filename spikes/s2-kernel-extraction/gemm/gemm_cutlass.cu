// S2 spike (R-006): one CUTLASS bf16 GEMM instantiation per tile config, behind plain functions.
// Row-major A [M,K] times W [N,K]^T -> C [M,N], fp32 accumulate, bf16 in and out.
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#include "gemm_cutlass.h"

namespace {
using bf16 = cutlass::bfloat16_t;
using Epi = cutlass::epilogue::thread::LinearCombination<bf16, 8, float, float>;
using Swz = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

template <typename Gemm>
int Run(const void* a, const void* w, void* c, int m, int n, int k, void* ws, void* stream) {
  Gemm op;
  typename Gemm::Arguments args({m, n, k}, {(const bf16*)a, k}, {(const bf16*)w, k},
                                {(bf16*)c, n}, {(bf16*)c, n}, {1.f, 0.f});
  if (op.can_implement(args) != cutlass::Status::kSuccess) return 2;
  if (op.initialize(args, ws, (cudaStream_t)stream) != cutlass::Status::kSuccess) return 3;
  return op.run((cudaStream_t)stream) == cutlass::Status::kSuccess ? 0 : 4;
}

using Large = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor, bf16, cutlass::layout::ColumnMajor, bf16,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>, Epi, Swz, 3>;
using Small = cutlass::gemm::device::Gemm<
    bf16, cutlass::layout::RowMajor, bf16, cutlass::layout::ColumnMajor, bf16,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>, Epi, Swz, 4>;
}  // namespace

extern "C" {
int roost_s2_gemm_cutlass_128x128x32(const void* a, const void* w, void* c, int m, int n, int k,
                                     void* ws, void* stream) {
  return Run<Large>(a, w, c, m, n, k, ws, stream);
}
int roost_s2_gemm_cutlass_32x64x64(const void* a, const void* w, void* c, int m, int n, int k,
                                   void* ws, void* stream) {
  return Run<Small>(a, w, c, m, n, k, ws, stream);
}
}
