/*
 * roost_worker_shm.h: records that live in the shared-memory segment between the host and a worker process.
 *
 * Contract C2 (StepPlan and StepResult) and the C8 segment, ring and handshake records (build plan section 5),
 * frozen as contract v0 by R-004. Mirrored field for field by Starlings.Roost.Abstractions; the layout tests compare
 * both against tests/Starlings.Roost.Abstractions.Tests/Layout/abi-layout.golden.txt.
 *
 * Rules (the ADR, README.md "Contracts", gives the reasons):
 *   - The segment is mapped at different addresses in each process, so no field is a pointer, host or device.
 *     Every reference is a byte offset from the start of the buffer that holds it, an index into a table, or an
 *     opaque handle. Device buffers are named by slot or handle and resolved by the worker.
 *   - Fixed-width types only, explicit reserved fields instead of padding, no arrays: a variable-length list is a
 *     RoostTableRef (offset, count, stride) into the same buffer.
 *   - Every header carries magic, layout_version and header_size. A field is added by appending it and bumping
 *     ROOST_SHM_LAYOUT_VERSION; readers accept header_size larger than they know and ignore the tail.
 *   - Unused and reserved fields are zero; the worker asserts that until the feature that uses them lands.
 *   - Integers are little-endian (every supported host is).
 */
#ifndef ROOST_WORKER_SHM_H_
#define ROOST_WORKER_SHM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROOST_SHM_LAYOUT_VERSION 1

#define ROOST_SHM_MAGIC_SEGMENT 0x47455352u /* "RSEG" */
#define ROOST_SHM_MAGIC_PLAN 0x4C505352u    /* "RSPL" */
#define ROOST_SHM_MAGIC_RESULT 0x53525352u  /* "RSRS" */
#define ROOST_SHM_MAGIC_RING 0x474E5252u    /* "RRNG" */

/* Alignment of every table, ring cursor and buffer in the segment: one cache line. */
#define ROOST_SHM_ALIGNMENT 64

/* A table in the same buffer: byte offset from the buffer start, entry count, bytes per entry. */
typedef struct RoostTableRef {
  uint64_t offset;
  uint32_t count;
  uint32_t stride;
} RoostTableRef;

/* ==== C2: step plan ============================================================================================ */

#define ROOST_MATH_DEFAULT 0
#define ROOST_MATH_DETERMINISTIC 1 /* deterministic, batch-invariant variants only (C1 flags) */
#define ROOST_MATH_FP32_REFERENCE 2

#define ROOST_PLAN_FLAG_GRAPH 0x1u    /* this step may replay a captured graph for its shape bucket */
#define ROOST_PLAN_FLAG_PREFILL 0x2u  /* at least one sequence is prefilling */

typedef struct RoostStepPlanHeader {
  uint32_t magic;          /* ROOST_SHM_MAGIC_PLAN */
  uint32_t layout_version; /* ROOST_SHM_LAYOUT_VERSION */
  uint32_t header_size;
  uint32_t flags;          /* ROOST_PLAN_FLAG_* */
  uint64_t step_id;
  uint64_t total_bytes;    /* bytes of this buffer in use, header included */
  uint32_t shape_bucket;
  uint32_t math_mode;      /* ROOST_MATH_* */
  uint32_t sequence_count;
  uint32_t token_count;
  uint32_t group_count;    /* state groups in the profile (C4) */
  uint32_t rank;           /* reserved (C9): this plan's mesh rank, 0 on a single rank */
  uint32_t mesh_id;        /* reserved (C9) */
  uint32_t reserved0;
  RoostTableRef sequences;        /* RoostPlanSequence */
  RoostTableRef token_ids;        /* int32, token_count */
  RoostTableRef positions;        /* int32, token_count */
  RoostTableRef group_blocks;     /* RoostPlanGroupBlocks, sequence_count * group_count, sequence-major */
  RoostTableRef block_ids;        /* int32 block ids, referenced by group_blocks */
  RoostTableRef sampling;         /* RoostPlanSampling */
  RoostTableRef cache_ops;        /* RoostPlanCacheOp, executed before the forward */
  RoostTableRef embedding_inputs; /* RoostPlanEmbeddingInput */
  RoostTableRef stop_token_ids;   /* int32, referenced by RoostPlanSequence.stop_start and stop_count */
  /* Reserved from v0: zero count until the feature lands. */
  RoostTableRef spec_tree_parents; /* int32 parent index per draft node, -1 for the root (M2 speculation trees) */
  RoostTableRef spec_tree_masks;   /* uint64 attention mask words per draft node */
  RoostTableRef moe_expert_counts; /* int32 tokens per expert (M2) */
  RoostTableRef moe_expert_placement; /* int32 per expert: local device slot, or -1 for host experts (M2, bet X6) */
  RoostTableRef kv_transfers;      /* RoostPlanKvTransfer (M4) */
  RoostTableRef head_rows;         /* RoostPlanHeadRow: rows that end in a head instead of the sampler */
  RoostTableRef encoder_outputs;   /* RoostPlanEncoderOutput (M5 multimodal) */
  uint64_t reserved1;
  uint64_t reserved2;
} RoostStepPlanHeader;

#define ROOST_SEQ_FLAG_PREFILL 0x1u    /* this chunk is prefill */
#define ROOST_SEQ_FLAG_SAMPLE 0x2u     /* sample after this chunk (last chunk of a prompt, or decode) */
#define ROOST_SEQ_FLAG_VERIFY 0x4u     /* the last draft_count tokens are drafts to verify */
#define ROOST_SEQ_FLAG_HEAD_ROW 0x8u   /* reserved: ends in head_rows, not the sampler */

typedef struct RoostPlanSequence {
  uint64_t sequence_handle;  /* engine's id for the sequence; echoed in the result */
  uint32_t token_start;      /* index into token_ids and positions */
  uint32_t token_count;
  uint32_t position_offset;  /* position of the first token, i.e. tokens already in the cache */
  uint32_t group_blocks_start; /* index into group_blocks; group_count consecutive entries */
  int32_t recurrent_slot;    /* recurrent-state pool slot (C4), -1 for none */
  int32_t lora_slot;         /* -1 for the base model */
  int32_t grammar_slot;      /* token-mask slot, -1 for none */
  int32_t embedding_slot;    /* index into embedding_inputs, -1 for none */
  uint32_t sampling_index;   /* index into sampling */
  uint32_t draft_count;
  uint32_t flags;            /* ROOST_SEQ_FLAG_* */
  uint32_t stop_start;       /* index into stop_token_ids: the model's and the request's stop ids */
  uint32_t stop_count;
  uint32_t max_new_tokens_left; /* tokens the request may still emit; 0 means no limit */
  uint32_t spec_tree_start;  /* reserved: index into spec_tree_parents */
  uint32_t spec_tree_count;  /* reserved */
  int32_t head_row;          /* reserved: index into head_rows, -1 for none */
  int32_t encoder_output;    /* reserved: index into encoder_outputs, -1 for none */
} RoostPlanSequence;

typedef struct RoostPlanGroupBlocks {
  uint32_t block_start; /* index into block_ids */
  uint32_t block_count;
} RoostPlanGroupBlocks;

#define ROOST_SAMPLE_FLAG_GREEDY 0x1u
#define ROOST_SAMPLE_FLAG_SEEDED 0x2u

typedef struct RoostPlanSampling {
  float temperature;
  float top_p;
  int32_t top_k;            /* 0 disables */
  float min_p;
  float repetition_penalty; /* 1 disables */
  float presence_penalty;
  float frequency_penalty;
  uint32_t logprobs;        /* top-n logprobs to return, 0 for none */
  uint64_t seed;
  uint64_t philox_offset;
  uint32_t flags;           /* ROOST_SAMPLE_FLAG_* */
  uint32_t reserved0;
} RoostPlanSampling;

#define ROOST_CACHE_OP_COPY 1           /* block copy within a tier (copy-on-write) */
#define ROOST_CACHE_OP_TIER_MOVE 2      /* block move between tiers */
#define ROOST_CACHE_OP_SNAPSHOT_WRITE 3 /* recurrent state slot to snapshot pool */
#define ROOST_CACHE_OP_SNAPSHOT_RESTORE 4
#define ROOST_CACHE_OP_FORK 5           /* state-group fork: copy recurrent slot src to dst */

#define ROOST_TIER_DEVICE 0
#define ROOST_TIER_HOST 1
#define ROOST_TIER_DISK 2
#define ROOST_TIER_REMOTE 3

typedef struct RoostPlanCacheOp {
  uint32_t kind;     /* ROOST_CACHE_OP_* */
  uint32_t group;    /* state group index (C4) */
  uint32_t src;      /* first block or slot */
  uint32_t dst;
  uint32_t count;
  uint32_t src_tier; /* ROOST_TIER_* */
  uint32_t dst_tier;
  uint32_t reserved0;
} RoostPlanCacheOp;

/* Precomputed input embeddings that replace token embeddings for a token range. */
typedef struct RoostPlanEmbeddingInput {
  uint32_t sequence_index;
  uint32_t token_start;   /* index into token_ids */
  uint32_t token_count;
  uint32_t dtype;         /* ROOST_DTYPE_* */
  uint64_t buffer_handle; /* registered input region (C8), resolved by the worker */
  uint64_t byte_offset;   /* within that region */
} RoostPlanEmbeddingInput;

#define ROOST_KV_TRANSFER_SEND 1
#define ROOST_KV_TRANSFER_RECEIVE 2

/* Reserved (M4): KV blocks sent to or received from a peer. */
typedef struct RoostPlanKvTransfer {
  uint32_t kind;        /* ROOST_KV_TRANSFER_* */
  uint32_t peer_rank;
  uint32_t group;
  uint32_t block_start; /* index into block_ids */
  uint32_t block_count;
  uint32_t reserved0;
  uint64_t transfer_id;
} RoostPlanKvTransfer;

#define ROOST_HEAD_POOL_MEAN 1
#define ROOST_HEAD_POOL_LAST 2
#define ROOST_HEAD_POOL_CLS 3
#define ROOST_HEAD_DECISION 4 /* Kev pointer head (R-086) */
#define ROOST_HEAD_CLASSIFY 5

/* Reserved: pooling and decision rows. */
typedef struct RoostPlanHeadRow {
  uint32_t sequence_index;
  uint32_t head;          /* ROOST_HEAD_* */
  uint32_t token_start;   /* index into token_ids of the first row the head reads */
  uint32_t token_count;
  uint64_t output_offset; /* byte offset into the result's head_outputs region */
  uint32_t output_bytes;
  uint32_t reserved0;
} RoostPlanHeadRow;

/* Reserved (M5): encoder outputs (vision, audio) consumed by the decoder. */
typedef struct RoostPlanEncoderOutput {
  uint32_t sequence_index;
  uint32_t token_start;
  uint32_t token_count;
  uint32_t group;         /* encoder-output state group (C4) */
  uint64_t buffer_handle;
  uint64_t byte_offset;
} RoostPlanEncoderOutput;

/* ==== C2: step result ========================================================================================== */

/* Fault classes (S8). Recoverable and sticky are reported by the worker; hang, stalled and exited are concluded by
 * the host from the deadline, the heartbeat and the process. */
#define ROOST_FAULT_NONE 0
#define ROOST_FAULT_RECOVERABLE 1 /* the step failed; the worker and context are usable */
#define ROOST_FAULT_STICKY 2      /* the context is lost; the worker exits after publishing */
#define ROOST_FAULT_HANG 3        /* deadline missed, heartbeat alive */
#define ROOST_FAULT_STALLED 4     /* deadline missed, heartbeat stopped */
#define ROOST_FAULT_EXITED 5      /* worker process gone */

#define ROOST_FINISH_NONE 0
#define ROOST_FINISH_EOS 1        /* sampled a stop token id from the plan's stop table */
#define ROOST_FINISH_LENGTH 2
#define ROOST_FINISH_GRAMMAR 3    /* grammar matcher stopped */
#define ROOST_FINISH_ERROR 4

typedef struct RoostStepResultHeader {
  uint32_t magic;          /* ROOST_SHM_MAGIC_RESULT */
  uint32_t layout_version;
  uint32_t header_size;
  uint32_t fault;          /* ROOST_FAULT_* */
  uint64_t step_id;        /* the plan's step_id */
  uint64_t total_bytes;
  int32_t error_code;      /* CUDA or C1 status behind a fault, 0 otherwise */
  uint32_t sequence_count;
  uint32_t token_count;
  uint32_t reserved0;
  RoostTableRef sequences;    /* RoostResultSequence, plan order */
  RoostTableRef token_ids;    /* int32 emitted tokens */
  RoostTableRef logprobs;     /* RoostResultLogprob */
  RoostTableRef head_outputs; /* bytes, laid out by the plan's head_rows */
  RoostTableRef timings;      /* one RoostDeviceTimings */
  RoostTableRef expert_load;  /* reserved: int32 tokens routed per expert (M2) */
  RoostTableRef error_text;   /* UTF-8 bytes of the fault message, may be empty */
  uint64_t reserved1;
} RoostStepResultHeader;

typedef struct RoostResultSequence {
  uint64_t sequence_handle;
  uint32_t token_start;     /* index into token_ids */
  uint32_t token_count;     /* emitted: accepted drafts plus one */
  uint32_t accepted_drafts;
  uint32_t finish;          /* ROOST_FINISH_* */
  uint32_t logprob_start;   /* index into logprobs; (1 + top-n) rows per emitted token */
  uint32_t logprob_count;
} RoostResultSequence;

typedef struct RoostResultLogprob {
  int32_t token;
  float logprob;
} RoostResultLogprob;

typedef struct RoostDeviceTimings {
  uint64_t step_ns;    /* worker: plan seen to result published */
  uint64_t upload_ns;  /* plan tables to device */
  uint64_t forward_ns; /* GPU, from events */
  uint64_t sample_ns;
  uint64_t download_ns;
  uint32_t graph_replayed;
  uint32_t reserved0;
} RoostDeviceTimings;

/* ==== C8: segment, rings, handshake, heartbeat ================================================================= */

/* At offset 0 of the segment. Every *_offset is a byte offset from the segment start, aligned to
 * ROOST_SHM_ALIGNMENT. The worker registers the whole segment with cudaHostRegister. */
typedef struct RoostSegmentHeader {
  uint32_t magic;          /* ROOST_SHM_MAGIC_SEGMENT */
  uint32_t layout_version;
  uint32_t header_size;
  uint32_t contract_version; /* Starlings.Roost.ContractVersion.Current */
  uint64_t segment_bytes;
  uint64_t handshake_offset;
  uint64_t heartbeat_offset;
  uint64_t plan_ring_offset;   /* RoostRingHeader */
  uint64_t result_ring_offset; /* RoostRingHeader */
  uint64_t plan_buffers_offset;
  uint64_t plan_buffer_bytes;
  uint64_t result_buffers_offset;
  uint64_t result_buffer_bytes;
  uint32_t plan_buffer_count;   /* at least 2: plans are double buffered (S8: two in flight) */
  uint32_t result_buffer_count;
  uint64_t input_region_offset; /* region for embedding and encoder inputs, addressed by buffer_handle */
  uint64_t input_region_bytes;
  uint64_t reserved0;
} RoostSegmentHeader;

/* Single-producer single-consumer ring. Cursors are monotonically increasing uint64 slot counts, each alone on its
 * own cache line at the given offsets; slot i lives at slots_offset + (i % slot_count) * slot_bytes. */
typedef struct RoostRingHeader {
  uint32_t magic;          /* ROOST_SHM_MAGIC_RING */
  uint32_t layout_version;
  uint32_t slot_count;
  uint32_t slot_bytes;
  uint64_t slots_offset;
  uint64_t producer_cursor_offset;
  uint64_t consumer_cursor_offset;
  uint64_t reserved0;
} RoostRingHeader;

#define ROOST_RING_SLOT_STEP 1
#define ROOST_RING_SLOT_SHUTDOWN 2
#define ROOST_RING_SLOT_DRAIN 3

/* One ring slot: names a plan or result buffer by index. */
typedef struct RoostRingSlot {
  uint64_t step_id;
  uint32_t kind;         /* ROOST_RING_SLOT_* */
  uint32_t buffer_index; /* plan or result buffer */
  uint64_t deadline_ns;  /* host monotonic clock; 0 means none */
  uint64_t published_ns;
} RoostRingSlot;

#define ROOST_HANDSHAKE_EMPTY 0
#define ROOST_HANDSHAKE_OFFERED 1  /* host wrote its side */
#define ROOST_HANDSHAKE_READY 2    /* worker accepted: context, segment registration, weights done */
#define ROOST_HANDSHAKE_REJECTED 3

#define ROOST_REJECT_NONE 0
#define ROOST_REJECT_CONTRACT 1
#define ROOST_REJECT_ABI 2
#define ROOST_REJECT_MANIFEST_DIGEST 3
#define ROOST_REJECT_WEIGHTS_DIGEST 4
#define ROOST_REJECT_DEVICE 5
#define ROOST_REJECT_MESH 6
#define ROOST_REJECT_LAYOUT 7

/* Digests are SHA-256 in four little-endian words. The weights digest is the one recorded in the manifest and
 * profile lock; a worker never hashes weights at spawn (S8: 9.4 s for 3.44 GB). */
typedef struct RoostHandshake {
  uint32_t state;            /* ROOST_HANDSHAKE_* */
  uint32_t reject_reason;    /* ROOST_REJECT_* */
  uint32_t contract_version;
  uint32_t abi_version;      /* C1 (major << 16) | minor */
  uint32_t layout_version;   /* ROOST_SHM_LAYOUT_VERSION */
  int32_t device_ordinal;
  uint32_t mesh_rank;
  uint32_t mesh_size;
  uint64_t manifest_digest0;
  uint64_t manifest_digest1;
  uint64_t manifest_digest2;
  uint64_t manifest_digest3;
  uint64_t weights_digest0;
  uint64_t weights_digest1;
  uint64_t weights_digest2;
  uint64_t weights_digest3;
  uint64_t device_uuid0;     /* written by the worker */
  uint64_t device_uuid1;
  uint64_t host_pid;
  uint64_t worker_pid;
  uint64_t step_deadline_ns; /* per-step deadline the host enforces */
  uint64_t heartbeat_period_ns;
} RoostHandshake;

#define ROOST_WORKER_STARTING 0
#define ROOST_WORKER_READY 1
#define ROOST_WORKER_STEPPING 2
#define ROOST_WORKER_FAULTED 3
#define ROOST_WORKER_EXITING 4

/* Written by the worker's heartbeat thread; the host reads it to tell a hang from a stall. */
typedef struct RoostHeartbeat {
  uint64_t counter;
  uint64_t worker_ns;    /* worker monotonic clock at the last beat */
  uint64_t last_step_id; /* last step whose result was published */
  uint32_t state;        /* ROOST_WORKER_* */
  uint32_t reserved0;
} RoostHeartbeat;

#ifdef __cplusplus
}
#endif

#endif /* ROOST_WORKER_SHM_H_ */
