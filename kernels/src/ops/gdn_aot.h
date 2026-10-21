// The Triton AOT kernels of GatedDeltaRule (R-034), as build_gdn.py embeds them. One table per SM, generated into the
// build tree (never committed): cubin bytes, kernel name, run-time argument list, shared-memory size and thread count.
#ifndef ROOST_GDN_AOT_H_
#define ROOST_GDN_AOT_H_

#include <cstdint>

namespace roost_gdn {

// Table order; kernels/aot/build_gdn.py has the same list (ROLES).
enum GdnRole : int {
  kCumsum = 0,
  kKktSolve,
  kRecomputeWU,
  kChunkHInit,  // chunk state kernel with an initial state
  kChunkH,      // without
  kChunkO,
  kRecurrent,
  kGdnKernelCount
};

struct GdnKernel {
  const char* role;
  const char* kernel_name;
  const char* signature;  // "name:type,..." of the run-time arguments, in launch order
  const unsigned char* cubin;
  uint64_t cubin_bytes;
  uint32_t shared_bytes;  // dynamic shared memory per block
  uint32_t threads;
  const char* cubin_sha256;
};

extern const GdnKernel kGdnKernels[kGdnKernelCount];

}  // namespace roost_gdn

#endif  // ROOST_GDN_AOT_H_
