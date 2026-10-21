// Gemm and LmHead (ROOST_OP_GEMM, ROOST_OP_LM_HEAD), R-017: cuBLASLt with the algorithm pinned per shape bucket (S2).
//
// S2 measured that cuBLASLt's heuristic changes the split-K count with M, so rows differ bitwise between batch sizes, and
// that pinning one algorithm per (N, K, dtype) bucket restores bitwise equality across M. plan() therefore asks the
// heuristic once, for the bucket's M (RoostGemmParams::shape_bucket, 0 meaning the call's own M), and records the chosen
// cublasLtMatmulAlgo_t in ws->plan. run() only rebuilds the (stack) descriptors and calls cublasLtMatmul with that
// algorithm: no heap allocation and no heuristic query inside a run. The descriptors use cuBLASLt's opaque-init API on
// caller storage, so nothing is created or destroyed per call.
//
// Row-major to column-major: a row-major [r, c] matrix with row stride ld is the column-major [c, r] matrix with leading
// dimension ld. D[m,n] = A[m,k] B[k,n] is then D_c[n,m] = B_c[n,k] A_c[k,m], so cuBLASLt's "A" is our B and its "B" is our A.
//
// Specialisation: bf16 or f16 A, B and D (fp32 accumulation, CUBLAS_COMPUTE_32F, no TF32 path since the inputs are 16 bit),
// A not transposed, B stored [k, n] or transposed [n, k]. LmHead takes 16-bit hidden and weight and writes fp32 logits.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <cstring>
#include <type_traits>

#include "op_common.h"

namespace roost_gemm {

using namespace roost_ops;

constexpr uint32_t kPlanMagic = 0x52474d31;  // "RGM1"
// Scratch for split-K and stream-K reductions. Fixed, so the workspace query depends only on the op.
constexpr uint64_t kCublasWorkspaceBytes = 8ull << 20;
constexpr uint32_t kWorkspaceAlignment = 256;

const char* LtStatusName(cublasStatus_t st) {
  switch (st) {
    case CUBLAS_STATUS_SUCCESS: return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED: return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE: return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "architecture mismatch";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "not supported";
    default: return "unknown status";
  }
}

RoostStatus FromLt(cublasStatus_t st, const char* what) {
  if (st == CUBLAS_STATUS_SUCCESS) return ROOST_STATUS_OK;
  const RoostStatus mapped = st == CUBLAS_STATUS_NOT_SUPPORTED   ? ROOST_STATUS_UNSUPPORTED
                             : st == CUBLAS_STATUS_ALLOC_FAILED ? ROOST_STATUS_OUT_OF_MEMORY
                             : st == CUBLAS_STATUS_INVALID_VALUE ? ROOST_STATUS_INVALID_ARGUMENT
                                                                 : ROOST_STATUS_INTERNAL;
  return Fail(mapped, std::string("cuBLASLt ") + what + ": " + LtStatusName(st));
}

// What plan() leaves for run(). Trivially copyable; read with memcpy.
struct PlanRecord {
  uint32_t magic;
  uint32_t op;
  uint64_t n;
  uint64_t k;
  uint64_t bucket_m;  // the M the algorithm was chosen for
  uint32_t in_dtype;
  uint32_t out_dtype;
  uint32_t transpose_b;
  uint32_t has_bias;
  uint64_t workspace_bytes;  // cuBLASLt scratch the algorithm needs
  int32_t algo_id;
  int32_t tile_id;
  int32_t stages_id;
  int32_t split_k;
  cublasLtMatmulAlgo_t algo;
};
static_assert(std::is_trivially_copyable_v<PlanRecord>, "the plan record is copied as bytes");

cudaDataType ToCudaType(uint32_t dtype) {
  switch (dtype) {
    case ROOST_DTYPE_BF16: return CUDA_R_16BF;
    case ROOST_DTYPE_F16: return CUDA_R_16F;
    case ROOST_DTYPE_F32: return CUDA_R_32F;
    default: return CUDA_R_32F;
  }
}

// One matrix multiplication in our row-major terms.
struct Mm {
  uint64_t m, n, k;
  uint64_t lda, ldb, ldd, ldc;
  uint32_t in_dtype;   // A and B
  uint32_t out_dtype;  // C and D
  bool transpose_b;
  bool has_bias;
  const void* a;
  const void* b;
  const void* c;  // equals d when beta is zero
  void* d;
  const void* bias;
  float alpha;
  float beta;
};

// The descriptors of one multiplication, built in caller storage.
struct Descriptors {
  cublasLtMatmulDescOpaque_t desc_storage;
  cublasLtMatrixLayoutOpaque_t a_storage, b_storage, c_storage, d_storage;
  cublasLtMatmulPreferenceOpaque_t pref_storage;
  cublasLtMatmulDesc_t desc = &desc_storage;
  cublasLtMatrixLayout_t la = &a_storage, lb = &b_storage, lc = &c_storage, ld = &d_storage;
  cublasLtMatmulPreference_t pref = &pref_storage;
};

RoostStatus BuildDescriptors(const Mm& mm, uint64_t m_for_layout, Descriptors* ds) {
  RoostStatus s = FromLt(cublasLtMatmulDescInit(ds->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc init");
  if (s) return s;
  const cublasOperation_t op_n = CUBLAS_OP_N, op_t = CUBLAS_OP_T;
  // cuBLASLt "A" is our B; its "B" is our A (see the file comment).
  const cublasOperation_t op_a = mm.transpose_b ? op_t : op_n;
  s = FromLt(cublasLtMatmulDescSetAttribute(ds->desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)), "TRANSA");
  if (s) return s;
  s = FromLt(cublasLtMatmulDescSetAttribute(ds->desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)), "TRANSB");
  if (s) return s;
  if (mm.has_bias) {
    const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    const cudaDataType bias_type = ToCudaType(mm.out_dtype);
    s = FromLt(cublasLtMatmulDescSetAttribute(ds->desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)), "EPILOGUE");
    if (s) return s;
    s = FromLt(cublasLtMatmulDescSetAttribute(ds->desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                              sizeof(bias_type)),
               "BIAS_DATA_TYPE");
    if (s) return s;
    const void* bias = mm.bias;
    s = FromLt(cublasLtMatmulDescSetAttribute(ds->desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)),
               "BIAS_POINTER");
    if (s) return s;
  }
  const cudaDataType in_t = ToCudaType(mm.in_dtype), out_t = ToCudaType(mm.out_dtype);
  // cuBLASLt A (= our B): [n, k] column-major, or [k, n] when transposed storage is used (row-major [n,k] -> col [k,n]).
  s = FromLt(mm.transpose_b ? cublasLtMatrixLayoutInit(ds->la, in_t, mm.k, mm.n, mm.ldb)
                            : cublasLtMatrixLayoutInit(ds->la, in_t, mm.n, mm.k, mm.ldb),
             "layout A");
  if (s) return s;
  s = FromLt(cublasLtMatrixLayoutInit(ds->lb, in_t, mm.k, m_for_layout, mm.lda), "layout B");
  if (s) return s;
  s = FromLt(cublasLtMatrixLayoutInit(ds->lc, out_t, mm.n, m_for_layout, mm.ldc), "layout C");
  if (s) return s;
  return FromLt(cublasLtMatrixLayoutInit(ds->ld, out_t, mm.n, m_for_layout, mm.ldd), "layout D");
}

// Chooses the algorithm for `mm` at M = bucket_m and fills the plan record.
RoostStatus PlanMm(RoostContext* ctx, const Mm& mm, uint64_t bucket_m, uint64_t workspace_cap, uint32_t op,
                   PlanRecord* rec) {
  Descriptors ds;
  if (auto s = BuildDescriptors(mm, bucket_m, &ds)) return s;
  if (auto s = FromLt(cublasLtMatmulPreferenceInit(ds.pref), "preference init")) return s;
  const uint64_t cap = workspace_cap;
  if (auto s = FromLt(cublasLtMatmulPreferenceSetAttribute(ds.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cap,
                                                           sizeof(cap)),
                      "preference workspace")) {
    return s;
  }
  // Split-K may keep its partial sums in the output type (CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE) or accumulate them into the
  // output buffer; with a 16-bit output both lose the fp32 accumulation C1 promises (the first S2-style comparison with CpuOps
  // showed errors of several bf16 ulps for split-K plans). Only out-of-place reduction in the compute type is allowed.
  const uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  if (auto s = FromLt(cublasLtMatmulPreferenceSetAttribute(ds.pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                                           sizeof(reduction)),
                      "preference reduction scheme")) {
    return s;
  }
  // Alignment of the pointers the run will use, so that the chosen algorithm is valid for them.
  auto align_of = [](const void* p, uint64_t ld_elems, uint32_t eb) {
    uint32_t a = 16;
    while (a > 1 && (!Aligned(p, a) || ((ld_elems * eb) % a) != 0)) a /= 2;
    return a;
  };
  const uint32_t ib = DtypeBytes(mm.in_dtype), ob = DtypeBytes(mm.out_dtype);
  const uint32_t al_a = align_of(mm.b, mm.ldb, ib), al_b = align_of(mm.a, mm.lda, ib);
  const uint32_t al_c = align_of(mm.c, mm.ldc, ob), al_d = align_of(mm.d, mm.ldd, ob);
  cublasLtMatmulPreferenceSetAttribute(ds.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &al_a, sizeof(al_a));
  cublasLtMatmulPreferenceSetAttribute(ds.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &al_b, sizeof(al_b));
  cublasLtMatmulPreferenceSetAttribute(ds.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &al_c, sizeof(al_c));
  cublasLtMatmulPreferenceSetAttribute(ds.pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &al_d, sizeof(al_d));
  cublasLtMatmulHeuristicResult_t result;
  int found = 0;
  if (auto s = FromLt(cublasLtMatmulAlgoGetHeuristic(static_cast<cublasLtHandle_t>(ctx->cublaslt), ds.desc, ds.la,
                                                     ds.lb, ds.lc, ds.ld, ds.pref, 1, &result, &found),
                      "heuristic")) {
    return s;
  }
  if (found < 1) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "cuBLASLt returned no algorithm for this shape");
  }
  std::memset(rec, 0, sizeof(*rec));
  rec->magic = kPlanMagic;
  rec->op = op;
  rec->n = mm.n;
  rec->k = mm.k;
  rec->bucket_m = bucket_m;
  rec->in_dtype = mm.in_dtype;
  rec->out_dtype = mm.out_dtype;
  rec->transpose_b = mm.transpose_b;
  rec->has_bias = mm.has_bias;
  rec->workspace_bytes = result.workspaceSize;
  rec->algo = result.algo;
  int32_t v = 0;
  size_t written = 0;
  if (cublasLtMatmulAlgoConfigGetAttribute(&result.algo, CUBLASLT_ALGO_CONFIG_ID, &v, sizeof(v), &written) ==
      CUBLAS_STATUS_SUCCESS) {
    rec->algo_id = v;
  }
  if (cublasLtMatmulAlgoConfigGetAttribute(&result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &v, sizeof(v), &written) ==
      CUBLAS_STATUS_SUCCESS) {
    rec->tile_id = v;
  }
  if (cublasLtMatmulAlgoConfigGetAttribute(&result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &v, sizeof(v), &written) ==
      CUBLAS_STATUS_SUCCESS) {
    rec->stages_id = v;
  }
  if (cublasLtMatmulAlgoConfigGetAttribute(&result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v, sizeof(v), &written) ==
      CUBLAS_STATUS_SUCCESS) {
    rec->split_k = v;
  }
  return ROOST_STATUS_OK;
}

RoostStatus RunMm(RoostContext* ctx, const Mm& mm, const PlanRecord& rec, void* workspace, uint64_t workspace_bytes,
                  void* stream) {
  if (rec.workspace_bytes > workspace_bytes) {
    return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "cuBLASLt scratch is larger than the device workspace");
  }
  Descriptors ds;
  if (auto s = BuildDescriptors(mm, mm.m, &ds)) return s;
  const cublasStatus_t st = cublasLtMatmul(static_cast<cublasLtHandle_t>(ctx->cublaslt), ds.desc, &mm.alpha, mm.b, ds.la,
                                           mm.a, ds.lb, &mm.beta, mm.c, ds.lc, mm.d, ds.ld, &rec.algo, workspace,
                                           rec.workspace_bytes, static_cast<cudaStream_t>(stream));
  return FromLt(st, "matmul");
}

// ---- Gemm ----------------------------------------------------------------------------------------------------------

RoostStatus ValidateGemm(const RoostGemmParams& p, Mm* mm) {
  constexpr char kOp[] = "Gemm";
  if (p.reserved0 != 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: reserved0 must be zero");
  if (p.transpose_a != 0) return Fail(ROOST_STATUS_UNSUPPORTED, "Gemm: transpose_a is not built");
  if (p.a.dtype != ROOST_DTYPE_BF16 && p.a.dtype != ROOST_DTYPE_F16) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "Gemm: a must be bf16 or f16");
  }
  if (p.m == 0 || p.n == 0 || p.k == 0) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: m, n and k must be non-zero");
  if (p.m > INT32_MAX || p.n > INT32_MAX || p.k > INT32_MAX) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "Gemm: a dimension exceeds 2^31 - 1");
  }
  const uint64_t lda = p.lda == 0 ? p.k : p.lda;
  const uint64_t ldb = p.ldb == 0 ? (p.transpose_b ? p.k : p.n) : p.ldb;
  const uint64_t ldd = p.ldd == 0 ? p.n : p.ldd;
  if (lda < p.k || ldb < (p.transpose_b ? p.k : p.n) || ldd < p.n) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: a leading dimension is smaller than the row width");
  }
  const bool use_c = p.beta != 0.0f;
  const bool bias = p.bias.address != nullptr;
  const uint32_t dt = p.a.dtype;
  // R-019 finding: for bf16 the pinned algorithms exceed the gemm-bf16 class (1.09 to 1.95 of the limit, 44 elements in 5 conformance
  // cases) when a C input (beta != 0) and a bias are both present; f16 and every other combination are within it. No generated forward
  // (Qwen3.5-0.8B, roost-tiny, roost-tiny-hybrid) uses the combination, so it is refused rather than served out of class. The check sits in
  // ValidateGemm so plan() and run() both return the typed ROOST_STATUS_UNSUPPORTED before any launch. RoostConstraints has no field for an
  // epilogue (a C1 change, stream A), hence no manifest constraint; kernels/README.md records the kernel-fix follow-up.
  if (dt == ROOST_DTYPE_BF16 && use_c && bias) {
    return Fail(ROOST_STATUS_UNSUPPORTED,
                "Gemm: bf16 with both a C input (beta != 0) and a bias is outside the gemm-bf16 tolerance class and is refused");
  }
  const uint64_t a_len = (p.m - 1) * lda + p.k;
  const uint64_t b_rows = p.transpose_b ? p.n : p.k;
  const uint64_t b_len = (b_rows - 1) * ldb + (p.transpose_b ? p.k : p.n);
  const uint64_t d_len = (p.m - 1) * ldd + p.n;
  if (auto s = CheckSpan(p.a, dt, a_len, kOp, "a")) return s;
  if (auto s = CheckSpan(p.b, dt, b_len, kOp, "b")) return s;
  if (auto s = CheckSpan(p.d, dt, d_len, kOp, "d")) return s;
  if (use_c) {
    if (auto s = CheckSpan(p.c, dt, d_len, kOp, "c")) return s;
  }
  if (bias) {
    if (auto s = CheckSpan(p.bias, dt, p.n, kOp, "bias")) return s;
  }
  // The algorithm is planned for 16-byte alignment (the best cuBLASLt kernels need it); smaller alignment is refused so a
  // plan stays valid for every call of its bucket.
  const uint32_t es = DtypeBytes(dt);
  if (!Aligned(p.a.address, 16) || !Aligned(p.b.address, 16) || !Aligned(p.d.address, 16) ||
      (use_c && !Aligned(p.c.address, 16)) || (bias && !Aligned(p.bias.address, 16)) || (lda * es) % 16 != 0 ||
      (ldb * es) % 16 != 0 || (ldd * es) % 16 != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT,
                "Gemm: buffers and leading dimensions must be 16-byte aligned (multiples of 8 elements)");
  }
  mm->m = p.m;
  mm->n = p.n;
  mm->k = p.k;
  mm->lda = lda;
  mm->ldb = ldb;
  mm->ldd = ldd;
  mm->ldc = ldd;
  mm->in_dtype = dt;
  mm->out_dtype = dt;
  mm->transpose_b = p.transpose_b != 0;
  mm->has_bias = bias;
  mm->a = p.a.address;
  mm->b = p.b.address;
  mm->c = use_c ? p.c.address : p.d.address;
  mm->d = p.d.address;
  mm->bias = p.bias.address;
  mm->alpha = p.alpha;
  mm->beta = p.beta;
  return ROOST_STATUS_OK;
}

}  // namespace roost_gemm

extern "C" RoostStatus roost_gemm_workspace(RoostContext*, const void* params, RoostWorkspaceSizes* sizes) {
  using namespace roost_gemm;
  if (auto s = CheckHeader(params, sizeof(RoostGemmParams), "Gemm")) return s;
  if (sizes == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: sizes is NULL");
  sizes->device_bytes = kCublasWorkspaceBytes;
  sizes->pinned_host_bytes = 0;
  sizes->plan_bytes = sizeof(PlanRecord);
  sizes->device_alignment = kWorkspaceAlignment;
  return ROOST_STATUS_OK;
}

extern "C" RoostStatus roost_gemm_plan(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void*) {
  using namespace roost_gemm;
  if (ctx == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: ctx is NULL");
  if (auto s = CheckHeader(params, sizeof(RoostGemmParams), "Gemm")) return s;
  const auto& p = *static_cast<const RoostGemmParams*>(params);
  return Guarded("Gemm", [&]() -> RoostStatus {
    if (ws == nullptr || ws->plan == nullptr || ws->plan_bytes < sizeof(PlanRecord)) {
      return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "Gemm: plan needs a plan record of at least plan_bytes");
    }
    Mm mm{};
    if (auto s = ValidateGemm(p, &mm)) return s;
    const uint64_t bucket = p.shape_bucket != 0 ? p.shape_bucket : p.m;
    if (bucket < p.m) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: shape_bucket is smaller than m");
    PlanRecord rec;
    const uint64_t cap = ws->device_bytes < kCublasWorkspaceBytes ? ws->device_bytes : kCublasWorkspaceBytes;
    if (auto s = PlanMm(ctx, mm, bucket, cap, ROOST_OP_GEMM, &rec)) return s;
    std::memcpy(ws->plan, &rec, sizeof(rec));
    return ROOST_STATUS_OK;
  });
}

extern "C" RoostStatus roost_gemm_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream) {
  using namespace roost_gemm;
  if (ctx == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: ctx is NULL");
  if (auto s = CheckHeader(params, sizeof(RoostGemmParams), "Gemm")) return s;
  const auto& p = *static_cast<const RoostGemmParams*>(params);
  return Guarded("Gemm", [&]() -> RoostStatus {
    Mm mm{};
    if (auto s = ValidateGemm(p, &mm)) return s;
    if (ws == nullptr || ws->plan == nullptr || ws->plan_bytes < sizeof(PlanRecord)) return ROOST_STATUS_NOT_PLANNED;
    PlanRecord rec;
    std::memcpy(&rec, ws->plan, sizeof(rec));
    if (rec.magic != kPlanMagic || rec.op != ROOST_OP_GEMM) {
      roost_set_error("Gemm: no Gemm plan was made in this workspace");
      return ROOST_STATUS_NOT_PLANNED;
    }
    if (rec.n != mm.n || rec.k != mm.k || rec.in_dtype != mm.in_dtype || rec.transpose_b != mm.transpose_b ||
        rec.has_bias != mm.has_bias || mm.m > rec.bucket_m) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "Gemm: the plan was made for another shape bucket");
    }
    return RunMm(ctx, mm, rec, ws->device, ws->device_bytes, stream);
  });
}

// ---- LmHead --------------------------------------------------------------------------------------------------------

namespace roost_gemm {

// hidden rows -> contiguous scratch, one block per selected row. A row index outside [0, tokens) gathers zeros.
__global__ void GatherRowsKernel(const uint8_t* hidden, const int32_t* rows, uint8_t* out, uint32_t tokens,
                                 uint64_t row_bytes) {
  const int32_t r = rows[blockIdx.x];
  const bool ok = r >= 0 && static_cast<uint32_t>(r) < tokens;
  const uint4* src = reinterpret_cast<const uint4*>(hidden + static_cast<uint64_t>(ok ? r : 0) * row_bytes);
  uint4* dst = reinterpret_cast<uint4*>(out + static_cast<uint64_t>(blockIdx.x) * row_bytes);
  for (uint64_t i = threadIdx.x; i < row_bytes / 16; i += blockDim.x) {
    dst[i] = ok ? src[i] : make_uint4(0, 0, 0, 0);
  }
}

// logits = cap * tanh(logits / cap), in place, over the first `vocab` columns of each padded row.
__global__ void SoftCapKernel(float* logits, uint32_t vocab, uint32_t stride, float cap) {
  float* row = logits + static_cast<uint64_t>(blockIdx.y) * stride;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < vocab; i += gridDim.x * blockDim.x) {
    row[i] = cap * tanhf(row[i] / cap);
  }
}

struct LmHeadLayout {
  uint64_t scratch_bytes;  // gathered hidden rows
  uint64_t gemm_offset;    // start of cuBLASLt scratch
};

LmHeadLayout LayoutFor(const RoostLmHeadParams& p, uint32_t elem_bytes) {
  LmHeadLayout l;
  l.scratch_bytes = static_cast<uint64_t>(p.selected) * p.dim * elem_bytes;
  l.gemm_offset = AlignUp(l.scratch_bytes, kWorkspaceAlignment);
  return l;
}

RoostStatus ValidateLmHead(const RoostLmHeadParams& p, Mm* mm) {
  constexpr char kOp[] = "LmHead";
  if (p.hidden.dtype != ROOST_DTYPE_BF16 && p.hidden.dtype != ROOST_DTYPE_F16) {
    return Fail(ROOST_STATUS_UNSUPPORTED, "LmHead: hidden must be bf16 or f16");
  }
  if (p.selected == 0 || p.dim == 0 || p.vocab == 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: selected, dim and vocab must be non-zero");
  }
  if (p.vocab_padded < p.vocab) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: vocab_padded is smaller than vocab");
  if (p.dim % 8 != 0 || p.vocab_padded % 4 != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: dim must be a multiple of 8 and vocab_padded of 4");
  }
  const uint32_t dt = p.hidden.dtype;
  if (auto s = CheckSpan(p.hidden, dt, 1, kOp, "hidden")) return s;
  if (auto s = CheckSpan(p.rows, ROOST_DTYPE_I32, p.selected, kOp, "rows")) return s;
  if (auto s = CheckSpan(p.weight, dt, static_cast<uint64_t>(p.vocab) * p.dim, kOp, "weight")) return s;
  if (auto s = CheckSpan(p.logits, ROOST_DTYPE_F32, (p.selected - 1ull) * p.vocab_padded + p.vocab, kOp, "logits")) {
    return s;
  }
  if (!Aligned(p.hidden.address, 16) || !Aligned(p.weight.address, 16) || !Aligned(p.logits.address, 16)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: buffers must be 16-byte aligned");
  }
  mm->m = p.selected;
  mm->n = p.vocab;
  mm->k = p.dim;
  mm->lda = p.dim;
  mm->ldb = p.dim;
  mm->ldd = p.vocab_padded;
  mm->ldc = p.vocab_padded;
  mm->in_dtype = dt;
  mm->out_dtype = ROOST_DTYPE_F32;
  mm->transpose_b = true;
  mm->has_bias = false;
  mm->a = nullptr;  // the gathered scratch, set by the caller
  mm->b = p.weight.address;
  mm->c = p.logits.address;
  mm->d = p.logits.address;
  mm->bias = nullptr;
  mm->alpha = 1.0f;
  mm->beta = 0.0f;
  return ROOST_STATUS_OK;
}

}  // namespace roost_gemm

extern "C" RoostStatus roost_lm_head_workspace(RoostContext*, const void* params, RoostWorkspaceSizes* sizes) {
  using namespace roost_gemm;
  if (auto s = CheckHeader(params, sizeof(RoostLmHeadParams), "LmHead")) return s;
  if (sizes == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: sizes is NULL");
  const auto& p = *static_cast<const RoostLmHeadParams*>(params);
  const uint32_t eb = DtypeBytes(p.hidden.dtype);
  if (eb == 0) return Fail(ROOST_STATUS_UNSUPPORTED, "LmHead: hidden must be bf16 or f16");
  const LmHeadLayout l = LayoutFor(p, eb);
  sizes->device_bytes = l.gemm_offset + kCublasWorkspaceBytes;
  sizes->pinned_host_bytes = 0;
  sizes->plan_bytes = sizeof(PlanRecord);
  sizes->device_alignment = kWorkspaceAlignment;
  return ROOST_STATUS_OK;
}

extern "C" RoostStatus roost_lm_head_plan(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void*) {
  using namespace roost_gemm;
  if (ctx == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: ctx is NULL");
  if (auto s = CheckHeader(params, sizeof(RoostLmHeadParams), "LmHead")) return s;
  const auto& p = *static_cast<const RoostLmHeadParams*>(params);
  return Guarded("LmHead", [&]() -> RoostStatus {
    if (ws == nullptr || ws->plan == nullptr || ws->plan_bytes < sizeof(PlanRecord)) {
      return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "LmHead: plan needs a plan record of at least plan_bytes");
    }
    Mm mm{};
    if (auto s = ValidateLmHead(p, &mm)) return s;
    const uint64_t bucket = p.shape_bucket != 0 ? p.shape_bucket : p.selected;
    if (bucket < p.selected) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: shape_bucket is smaller than selected");
    mm.a = p.hidden.address;  // alignment stand-in; the run gathers into 16-byte aligned scratch
    PlanRecord rec;
    if (auto s = PlanMm(ctx, mm, bucket, kCublasWorkspaceBytes, ROOST_OP_LM_HEAD, &rec)) return s;
    std::memcpy(ws->plan, &rec, sizeof(rec));
    return ROOST_STATUS_OK;
  });
}

extern "C" RoostStatus roost_lm_head_run(RoostContext* ctx, const void* params, const RoostWorkspace* ws, void* stream) {
  using namespace roost_gemm;
  if (ctx == nullptr) return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: ctx is NULL");
  if (auto s = CheckHeader(params, sizeof(RoostLmHeadParams), "LmHead")) return s;
  const auto& p = *static_cast<const RoostLmHeadParams*>(params);
  return Guarded("LmHead", [&]() -> RoostStatus {
    Mm mm{};
    if (auto s = ValidateLmHead(p, &mm)) return s;
    if (ws == nullptr || ws->plan == nullptr || ws->plan_bytes < sizeof(PlanRecord) || ws->device == nullptr) {
      return ROOST_STATUS_NOT_PLANNED;
    }
    PlanRecord rec;
    std::memcpy(&rec, ws->plan, sizeof(rec));
    if (rec.magic != kPlanMagic || rec.op != ROOST_OP_LM_HEAD) {
      roost_set_error("LmHead: no LmHead plan was made in this workspace");
      return ROOST_STATUS_NOT_PLANNED;
    }
    if (rec.n != mm.n || rec.k != mm.k || rec.in_dtype != mm.in_dtype || mm.m > rec.bucket_m) {
      return Fail(ROOST_STATUS_INVALID_ARGUMENT, "LmHead: the plan was made for another shape bucket");
    }
    const uint32_t eb = DtypeBytes(mm.in_dtype);
    const LmHeadLayout l = LayoutFor(p, eb);
    if (ws->device_bytes < l.gemm_offset + rec.workspace_bytes || !Aligned(ws->device, 16)) {
      return Fail(ROOST_STATUS_WORKSPACE_TOO_SMALL, "LmHead: device workspace is smaller than the query reported");
    }
    const auto s = static_cast<cudaStream_t>(stream);
    GatherRowsKernel<<<p.selected, 256, 0, s>>>(static_cast<const uint8_t*>(p.hidden.address),
                                                static_cast<const int32_t*>(p.rows.address),
                                                static_cast<uint8_t*>(ws->device), static_cast<uint32_t>(p.hidden.length / p.dim),
                                                static_cast<uint64_t>(p.dim) * eb);
    if (auto st = FromCuda(cudaGetLastError())) return st;
    mm.a = ws->device;
    if (auto st = RunMm(ctx, mm, rec, static_cast<char*>(ws->device) + l.gemm_offset, ws->device_bytes - l.gemm_offset,
                        stream)) {
      return st;
    }
    if (p.soft_cap != 0.0f) {
      const dim3 grid((p.vocab + 255) / 256 < 1024 ? (p.vocab + 255) / 256 : 1024, p.selected);
      SoftCapKernel<<<grid, 256, 0, s>>>(static_cast<float*>(p.logits.address), p.vocab, p.vocab_padded, p.soft_cap);
      return FromCuda(cudaGetLastError());
    }
    return ROOST_STATUS_OK;
  });
}

// ---- Context -------------------------------------------------------------------------------------------------------

RoostStatus roost_gemm_context_init(RoostContext* ctx) {
  using namespace roost_gemm;
  cublasLtHandle_t handle = nullptr;
  const cublasStatus_t st = cublasLtCreate(&handle);
  if (st != CUBLAS_STATUS_SUCCESS) {
    return FromLt(st, "create");
  }
  ctx->cublaslt = handle;
  return ROOST_STATUS_OK;
}

void roost_gemm_context_release(RoostContext* ctx) {
  if (ctx != nullptr && ctx->cublaslt != nullptr) {
    cublasLtDestroy(static_cast<cublasLtHandle_t>(ctx->cublaslt));
    ctx->cublaslt = nullptr;
  }
}
