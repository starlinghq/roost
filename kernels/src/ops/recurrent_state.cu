// RecurrentStateCopy and RecurrentStateSnapshot (ROOST_OP_RECURRENT_STATE_COPY, ROOST_OP_RECURRENT_STATE_SNAPSHOT) for
// the recurrent pools of hybrid models (R-034). Both copy whole slots in pair order, so a later pair sees an earlier
// pair's write (CpuOps.RecurrentStateCopy and RecurrentStateSnapshot are the definition). Byte i of a destination slot
// depends only on byte i of a source slot, so every thread owns a range of byte offsets and walks all pairs in order:
// parallel across the slot, sequential across pairs, and exactly the CPU semantics.

#include <cuda_runtime.h>

#include <cstdint>

#include "hybrid_common.h"
#include "hybrid_params.h"

namespace roost_hybrid_state {

// Word is the unit one thread moves: 16, 4 or 1 bytes, chosen by the alignment of the slot size and both bases.
template <typename Word>
__global__ void CopyPairsKernel(const unsigned char* src_base, unsigned char* dst_base, const int32_t* pairs,
                                uint32_t count, uint64_t slot_bytes, uint32_t swap) {
  const uint64_t words = slot_bytes / sizeof(Word);
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  for (uint64_t w = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride) {
    for (uint32_t i = 0; i < count; ++i) {
      const uint64_t from = static_cast<uint64_t>(static_cast<uint32_t>(pairs[(2 * i) + swap])) * slot_bytes;
      const uint64_t to = static_cast<uint64_t>(static_cast<uint32_t>(pairs[(2 * i) + 1 - swap])) * slot_bytes;
      reinterpret_cast<Word*>(dst_base + to)[w] = reinterpret_cast<const Word*>(src_base + from)[w];
    }
  }
}

RoostStatus Launch(const void* src, void* dst, const RoostDeviceSpan& pairs, uint64_t slot_bytes, uint32_t count,
                   uint32_t swap, cudaStream_t stream) {
  using roost_hybrid::Aligned;
  if (count == 0 || slot_bytes == 0) {
    return ROOST_STATUS_OK;
  }
  const auto* s = static_cast<const unsigned char*>(src);
  auto* d = static_cast<unsigned char*>(dst);
  const auto* pr = static_cast<const int32_t*>(pairs.address);
  const bool wide = slot_bytes % 16 == 0 && Aligned(s, 16) && Aligned(d, 16);
  const bool word = slot_bytes % 4 == 0 && Aligned(s, 4) && Aligned(d, 4);
  const uint64_t unit = wide ? 16 : (word ? 4 : 1);
  const uint64_t words = slot_bytes / unit;
  constexpr uint32_t kThreads = 256;
  const uint64_t wanted = (words + kThreads - 1) / kThreads;
  const uint32_t blocks = static_cast<uint32_t>(wanted < 1024 ? wanted : 1024);
  if (wide) {
    CopyPairsKernel<uint4><<<blocks, kThreads, 0, stream>>>(s, d, pr, count, slot_bytes, swap);
  } else if (word) {
    CopyPairsKernel<uint32_t><<<blocks, kThreads, 0, stream>>>(s, d, pr, count, slot_bytes, swap);
  } else {
    CopyPairsKernel<unsigned char><<<blocks, kThreads, 0, stream>>>(s, d, pr, count, slot_bytes, swap);
  }
  const cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? ROOST_STATUS_OK : roost_hybrid::CudaFail(err);
}

bool PoolOk(const RoostDeviceSpan& s) { return s.reserved0 == 0 && s.address != nullptr; }

}  // namespace roost_hybrid_state

extern "C" RoostStatus roost_recurrent_state_copy_run(RoostContext* ctx, const void* params, const RoostWorkspace* /*ws*/,
                                                      void* stream) {
  using roost_hybrid::Fail;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RecurrentStateCopy: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostRecurrentStateCopyParams*>(params);
  if (p.struct_size < sizeof(RoostRecurrentStateCopyParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH, "RecurrentStateCopy: params struct_size or abi_minor is not supported by this library");
  }
  if (p.reserved0 != 0) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RecurrentStateCopy: reserved0 must be zero");
  }
  if (p.count == 0 || p.slot_bytes == 0) {
    return ROOST_STATUS_OK;
  }
  if (!roost_hybrid_state::PoolOk(p.pool) ||
      !roost_hybrid::SpanOk(p.pairs, ROOST_DTYPE_I32, uint64_t{p.count} * 2)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RecurrentStateCopy: pool is NULL, or pairs is not a long enough int32 span");
  }
  return roost_hybrid_state::Launch(p.pool.address, p.pool.address, p.pairs, p.slot_bytes, p.count, 0,
                                    static_cast<cudaStream_t>(stream));
}

extern "C" RoostStatus roost_recurrent_state_snapshot_run(RoostContext* ctx, const void* params,
                                                          const RoostWorkspace* /*ws*/, void* stream) {
  using roost_hybrid::Fail;
  if (ctx == nullptr || params == nullptr) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT, "RecurrentStateSnapshot: ctx or params is NULL");
  }
  const auto& p = *static_cast<const RoostRecurrentStateSnapshotParams*>(params);
  if (p.struct_size < sizeof(RoostRecurrentStateSnapshotParams) || p.abi_minor > ROOST_ABI_MINOR) {
    return Fail(ROOST_STATUS_ABI_MISMATCH,
                "RecurrentStateSnapshot: params struct_size or abi_minor is not supported by this library");
  }
  if (p.count == 0 || p.slot_bytes == 0) {
    return ROOST_STATUS_OK;
  }
  if (!roost_hybrid_state::PoolOk(p.pool) || !roost_hybrid_state::PoolOk(p.snapshots) ||
      !roost_hybrid::SpanOk(p.pairs, ROOST_DTYPE_I32, uint64_t{p.count} * 2)) {
    return Fail(ROOST_STATUS_INVALID_ARGUMENT,
                "RecurrentStateSnapshot: pool or snapshots is NULL, or pairs is not a long enough int32 span");
  }
  const bool restore = p.restore != 0;
  return roost_hybrid_state::Launch(restore ? p.snapshots.address : p.pool.address,
                                    restore ? p.pool.address : p.snapshots.address, p.pairs, p.slot_bytes, p.count,
                                    restore ? 1u : 0u, static_cast<cudaStream_t>(stream));
}
