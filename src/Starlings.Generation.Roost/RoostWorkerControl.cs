using System.Collections.Immutable;

namespace Starlings.Generation.Roost;

/// <summary>What <see cref="RoostChatClient.SleepAsync"/> releases (R-100).</summary>
public enum RoostSleepLevel
{
    /// <summary>The KV cache and recurrent state pools; the weights stay resident, so a wake only reallocates the pools.</summary>
    KvCache = 1,

    /// <summary>
    /// The state pools and the weights; a wake streams the weights again from their locked file and verifies them as read.
    /// (vLLM's level 2 also discards weights; its level 1 offloads them to host memory, which Roost does not do.)
    /// </summary>
    KvCacheAndWeights = 2,
}

/// <summary>A worker control operation (R-100).</summary>
public enum RoostWorkerOperation
{
    /// <summary><see cref="RoostChatClient.SleepAsync"/>.</summary>
    Sleep = 1,

    /// <summary><see cref="RoostChatClient.WakeUpAsync"/>.</summary>
    WakeUp = 2,

    /// <summary><see cref="RoostChatClient.UpdateWeightsAsync"/>.</summary>
    UpdateWeights = 3,
}

/// <summary>What a worker control operation did.</summary>
/// <param name="Operation">The operation.</param>
/// <param name="Duration">Host time from the operation reaching the driver thread to the worker's answer: for a wake, the wake time (the time from the call to a worker that can serve).</param>
/// <param name="WorkerDuration">The worker's own time for it, on its clock.</param>
/// <param name="Bytes">Sleep: bytes the worker released. Wake and weight update: bytes it allocated.</param>
/// <param name="IsSleeping">Whether the worker is asleep afterwards.</param>
/// <param name="WeightsDigest">The SHA-256 of the weights the worker now serves (verified as read); default while they are released.</param>
public sealed record RoostWorkerControlResult(
    RoostWorkerOperation Operation,
    TimeSpan Duration,
    TimeSpan WorkerDuration,
    long Bytes,
    bool IsSleeping,
    ImmutableArray<byte> WeightsDigest);
