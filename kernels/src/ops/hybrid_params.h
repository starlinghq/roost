// Params of the hybrid ops that roost_kernels.h (contract C1, stream A) does not declare yet: CausalConv1d,
// RecurrentStateCopy and RecurrentStateSnapshot. The field order mirrors the C# argument structs of C3
// (CausalConv1dArgs, RecurrentStateCopyArgs, RecurrentStateSnapshotArgs in src/Starlings.Roost.Ops/OpArgs.cs), with the
// header's conventions: struct_size and abi_minor first, fixed-width fields only, spans for buffers, reserved fields zero.
//
// PROVISIONAL. Contracts change only through stream A, so these live next to the kernels that use them until a stream A
// contract-change PR lifts them into roost_kernels.h (ABI 1.2) together with their C# mirrors and the abi-layout golden.
// Until then no loader can call these three ops: the C# side has no struct to marshal. R-034 reports the request.
#ifndef ROOST_HYBRID_PARAMS_H_
#define ROOST_HYBRID_PARAMS_H_

#include <stdint.h>

#include "roost_kernels.h"

/* Depthwise causal convolution over packed sequences, one filter per channel (CpuOps.CausalConv1d is the definition).
 * y_t = bias + sum_{j < width} weight[c, j] * x_{t - (width - 1) + j}, x before the sequence start coming from the slot's
 * conv_state (oldest first); weight[c, width - 1] multiplies the current token. SiLU is applied last when silu is set.
 * Afterwards the slot holds the last width - 1 inputs of [state, sequence]. The state is always read: a fresh sequence
 * needs a zeroed slot. input and output must not overlap. A sequence of zero tokens leaves its slot untouched. */
typedef struct RoostCausalConv1dParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan input;       /* [tokens, channels] */
  RoostDeviceSpan weight;      /* [channels, width] */
  RoostDeviceSpan bias;        /* optional [channels]; address NULL when unused */
  RoostDeviceSpan output;      /* [tokens, channels] */
  RoostDeviceSpan conv_state;  /* pool [slots, channels, width - 1] */
  RoostDeviceSpan state_slots; /* [batch] int32 */
  RoostDeviceSpan cu_seqlens;  /* [batch + 1] int32 */
  uint32_t batch;
  uint32_t tokens;
  uint32_t channels;
  uint32_t width;
  uint32_t silu;               /* nonzero: apply SiLU */
  uint32_t reserved0;
} RoostCausalConv1dParams;

/* Copies slot pairs[i, 0] to slot pairs[i, 1] of pool as slot_bytes bytes, in pair order: a later pair sees an earlier
 * pair's write. A pair with equal slots is a no-op. Slot indices are not range-checked (they are device data). */
typedef struct RoostRecurrentStateCopyParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan pool;
  RoostDeviceSpan pairs; /* [count, 2] int32 (source slot, destination slot) */
  uint64_t slot_bytes;
  uint32_t count;
  uint32_t reserved0;
} RoostRecurrentStateCopyParams;

/* pairs[i, 0] is a pool slot and pairs[i, 1] a snapshots entry. restore == 0 copies slot to snapshot, nonzero copies
 * snapshot to slot. Byte copies in pair order. */
typedef struct RoostRecurrentStateSnapshotParams {
  uint32_t struct_size;
  uint32_t abi_minor;
  RoostDeviceSpan pool;
  RoostDeviceSpan snapshots;
  RoostDeviceSpan pairs; /* [count, 2] int32 (slot, snapshot) */
  uint64_t slot_bytes;
  uint32_t count;
  uint32_t restore;
} RoostRecurrentStateSnapshotParams;

#endif /* ROOST_HYBRID_PARAMS_H_ */
