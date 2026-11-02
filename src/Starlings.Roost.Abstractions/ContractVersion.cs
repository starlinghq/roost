namespace Starlings.Roost;

/// <summary>
/// The version of the Roost contracts C1 to C9 (build plan section 5; the ADR is the "Contracts" section of
/// README.md).
/// </summary>
/// <remarks>
/// Version 1 is the first frozen set, the <c>v0</c> stage of the plan. Version 2 (2026-10-19) adds row strides for the
/// GatedDeltaRule q, k and v views (C3) and <c>RoostGatedDeltaRuleParams</c> (C1 ABI 1.1). Version 3 (2026-10-19) adds the
/// Qwen3.5 gating ops GatedDeltaRuleGates, GatedRmsNorm and SigmoidMul (C3) with their params, and the params of CausalConv1d,
/// RecurrentStateCopy and RecurrentStateSnapshot (C1 ABI 1.2). Version 4 (2026-11-01, R-042) extends SampleArgs (C3) and
/// <c>RoostSampleParams</c> (C1 ABI 1.3) with repetition, presence and frequency penalties over per-row token counts, a stop
/// flag, top-k logprobs and the sampling math mode. Version 5 (2026-11-01, R-032) adds async scheduling to C2: the plan
/// header's <c>input_step_id</c> and the sequence flag <c>ROOST_SEQ_FLAG_SAMPLED_INPUT</c>, a decode row whose input is the
/// previous step's sampled id (shared-memory layout version 2). Version 6 (2026-11-02, R-100) adds the C8 control messages
/// sleep, wake and replace weights (<c>ROOST_RING_SLOT_SLEEP</c>, <c>_WAKE</c>, <c>_REPLACE_WEIGHTS</c>, <c>RoostControlMessage</c>,
/// <c>RoostControlResult</c>) and the worker state <c>ROOST_WORKER_SLEEPING</c> (shared-memory layout version 3). Until the M0 gate (R-026) a contract may
/// still change through a stream A pull request; each such change raises <see cref="Current"/> and updates every
/// consumer in the same change. The M0 gate promotes the contracts to the <c>v1</c> stage.
/// </remarks>
public static class ContractVersion
{
    /// <summary>The current contract version, carried in the C8 handshake and segment header.</summary>
    public const int Current = 6;

    /// <summary>The plan's name for the current stage: <c>v0</c> until the M0 gate, then <c>v1</c>.</summary>
    public const string Stage = "v0";
}
