namespace Starlings.Roost.Plans;

/// <summary>Constants of the shared-memory layout (<c>kernels/include/roost_worker_shm.h</c>).</summary>
public static class SharedLayout
{
    /// <summary><c>ROOST_SHM_LAYOUT_VERSION</c>, carried by every header in the segment.</summary>
    public const uint Version = 1;

    /// <summary><c>ROOST_SHM_ALIGNMENT</c>: alignment of every table, cursor and buffer.</summary>
    public const int Alignment = 64;

    /// <summary><c>ROOST_SHM_MAGIC_SEGMENT</c>, "RSEG".</summary>
    public const uint MagicSegment = 0x47455352;

    /// <summary><c>ROOST_SHM_MAGIC_PLAN</c>, "RSPL".</summary>
    public const uint MagicPlan = 0x4C505352;

    /// <summary><c>ROOST_SHM_MAGIC_RESULT</c>, "RSRS".</summary>
    public const uint MagicResult = 0x53525352;

    /// <summary><c>ROOST_SHM_MAGIC_RING</c>, "RRNG".</summary>
    public const uint MagicRing = 0x474E5252;
}

/// <summary>Step plan flags (<c>ROOST_PLAN_FLAG_*</c>).</summary>
[Flags]
public enum PlanOptions : uint
{
    /// <summary>No flags.</summary>
    None = 0,

    /// <summary>The step may replay a captured graph for its shape bucket.</summary>
    Graph = 0x1,

    /// <summary>At least one sequence is prefilling.</summary>
    Prefill = 0x2,
}

/// <summary>Math modes (<c>ROOST_MATH_*</c>).</summary>
public enum MathMode : uint
{
    /// <summary>The profile's selected variants.</summary>
    Default = 0,

    /// <summary>Deterministic, batch-invariant variants only (C1 flags).</summary>
    Deterministic = 1,

    /// <summary>fp32 reference variants, for bisection.</summary>
    Fp32Reference = 2,
}

/// <summary>Per-sequence plan flags (<c>ROOST_SEQ_FLAG_*</c>).</summary>
[Flags]
public enum SequenceOptions : uint
{
    /// <summary>No flags.</summary>
    None = 0,

    /// <summary>This chunk is prefill.</summary>
    Prefill = 0x1,

    /// <summary>Sample after this chunk.</summary>
    Sample = 0x2,

    /// <summary>The last <c>DraftCount</c> tokens are drafts to verify.</summary>
    Verify = 0x4,

    /// <summary>Reserved: the sequence ends in a head row, not the sampler.</summary>
    HeadRow = 0x8,
}

/// <summary>Sampling flags (<c>ROOST_SAMPLE_FLAG_*</c>).</summary>
[Flags]
public enum SamplingOptions : uint
{
    /// <summary>No flags.</summary>
    None = 0,

    /// <summary>Argmax, ignoring the other parameters.</summary>
    Greedy = 0x1,

    /// <summary>The request gave a seed.</summary>
    Seeded = 0x2,
}

/// <summary>Cache op kinds (<c>ROOST_CACHE_OP_*</c>).</summary>
public enum CacheOpKind : uint
{
    /// <summary>Not an op.</summary>
    None = 0,

    /// <summary>Block copy within a tier (copy-on-write).</summary>
    Copy = 1,

    /// <summary>Block move between tiers.</summary>
    TierMove = 2,

    /// <summary>Recurrent state slot to the snapshot pool.</summary>
    SnapshotWrite = 3,

    /// <summary>Snapshot pool to a recurrent state slot.</summary>
    SnapshotRestore = 4,

    /// <summary>State-group fork of a recurrent slot.</summary>
    Fork = 5,
}

/// <summary>Storage tiers (<c>ROOST_TIER_*</c>), shared with C4.</summary>
public enum BlockTier : uint
{
    /// <summary>Device memory.</summary>
    Device = 0,

    /// <summary>Pinned host memory.</summary>
    Host = 1,

    /// <summary>Local disk.</summary>
    Disk = 2,

    /// <summary>Another host (M4).</summary>
    Remote = 3,
}

/// <summary>KV transfer directions (<c>ROOST_KV_TRANSFER_*</c>), reserved for M4.</summary>
public enum KvTransferKind : uint
{
    /// <summary>Not a transfer.</summary>
    None = 0,

    /// <summary>Send blocks to the peer.</summary>
    Send = 1,

    /// <summary>Receive blocks from the peer.</summary>
    Receive = 2,
}

/// <summary>Heads that end a row instead of the sampler (<c>ROOST_HEAD_*</c>), reserved.</summary>
public enum HeadKind : uint
{
    /// <summary>Not a head row.</summary>
    None = 0,

    /// <summary>Mean pooling.</summary>
    PoolMean = 1,

    /// <summary>Last-token pooling.</summary>
    PoolLast = 2,

    /// <summary>First-token pooling.</summary>
    PoolCls = 3,

    /// <summary>Kev decision (pointer) head (R-086).</summary>
    Decision = 4,

    /// <summary>Classification head.</summary>
    Classify = 5,
}

/// <summary>
/// Fault classes (<c>ROOST_FAULT_*</c>, S8). The worker reports recoverable and sticky faults in the step result;
/// the host concludes hang, stalled and exited from the deadline, the heartbeat and the process.
/// </summary>
public enum FaultClass : uint
{
    /// <summary>No fault.</summary>
    None = 0,

    /// <summary>The step failed; the worker and its context are usable.</summary>
    Recoverable = 1,

    /// <summary>The CUDA context is lost; the worker exits after publishing the result.</summary>
    Sticky = 2,

    /// <summary>Step deadline missed while the heartbeat advances.</summary>
    Hang = 3,

    /// <summary>Step deadline missed and the heartbeat stopped.</summary>
    Stalled = 4,

    /// <summary>The worker process exited.</summary>
    Exited = 5,
}

/// <summary>Why a sequence stopped in this step (<c>ROOST_FINISH_*</c>).</summary>
public enum FinishReason : uint
{
    /// <summary>Still running.</summary>
    None = 0,

    /// <summary>Sampled one of the plan's stop token ids.</summary>
    Eos = 1,

    /// <summary>Reached its token limit.</summary>
    Length = 2,

    /// <summary>The grammar matcher stopped.</summary>
    Grammar = 3,

    /// <summary>The step failed for this sequence.</summary>
    Error = 4,
}
