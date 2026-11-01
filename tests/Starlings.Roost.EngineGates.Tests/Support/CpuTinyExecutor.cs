using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Security.Cryptography;
using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.State;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.WorkerHost;
using HostOutcome = Starlings.Roost.WorkerHost.StepOutcome;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>
/// An in-process worker that runs a tiny model on CpuOps behind the real chat client: it reads each C2 plan with the eager
/// worker's <see cref="StepPlanReader"/>, runs <see cref="TinyCpuModel.Forward"/>, samples greedily with
/// <see cref="TinyCpuModel.Argmax"/> and writes the C2 result with the eager worker's <see cref="StepResultWriter"/>. Everything
/// from the chat client to the plan is therefore the GPU path's code; only the forward is the CPU model.
/// </summary>
internal sealed class CpuTinyExecutor : IStepExecutor
{
    /// <summary>The digest the profile names as its kernel manifest (no kernel library is loaded on the CPU).</summary>
    public static readonly ImmutableArray<byte> ManifestDigest = [.. SHA256.HashData(Encoding.UTF8.GetBytes("roost-engine-gates-cpuops"))];

    private readonly TinyProfile _profile;
    private readonly PlanLimits _limits;
    private TinyCpuModel? _model;

    public CpuTinyExecutor(TinyProfile profile, PlanLimits limits)
    {
        _profile = profile;
        _limits = limits;
    }

    /// <summary>Tokens emitted per sequence handle, in order.</summary>
    public ConcurrentDictionary<ulong, List<int>> Emitted { get; } = new();

    /// <summary>Plans the reader refused (must stay empty).</summary>
    public ConcurrentQueue<string> Rejections { get; } = new();

    public WorkerIdentity Identity => new(KernelAbi.Version, ManifestDigest, TinyCpuModel.WeightsDigest(_profile));

    /// <summary>The plan limits of a worker with the given pools, for <paramref name="profile"/>.</summary>
    /// <param name="profile">The model.</param>
    /// <param name="attentionBlocks">Attention pool blocks.</param>
    /// <param name="recurrentSlots">Recurrent pool slots.</param>
    /// <param name="maxTokens">Tokens per step.</param>
    /// <param name="maxSequences">Rows per step.</param>
    /// <returns>The limits.</returns>
    public static PlanLimits LimitsFor(TinyProfile profile, int attentionBlocks, int recurrentSlots, int maxTokens, int maxSequences)
    {
        var groups = TinyCpuModel.GroupsOf(profile);
        var attention = groups.First(g => g.Kind != StateGroupKind.Recurrent);
        bool recurrent = groups.Any(g => g.Kind == StateGroupKind.Recurrent);
        return new PlanLimits(maxTokens, maxSequences, attentionBlocks, maxSequences, TinyCpuModel.Vocab, attention.BlockSize, attentionBlocks,
            recurrent ? recurrentSlots : 0, attention.Index, groups.Length);
    }

    public HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid)
    {
        deviceUuid.Fill(0xC9);
        return deviceOrdinal == 0 ? HandshakeReject.None : HandshakeReject.Device;
    }

    public void RegisterSegment(nint address, long bytes)
    {
    }

    public HandshakeReject LoadWeights()
    {
        _model = new TinyCpuModel(_profile, _limits.AttentionBlocks, Math.Max(_limits.RecurrentSlots, 1), 0, _limits.MaxTokens, _limits.MaxSequences);
        return HandshakeReject.None;
    }

    public HostOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result)
    {
        var model = _model ?? throw new InvalidOperationException("the weights are not loaded");
        ParsedPlan parsed;
        try
        {
            parsed = StepPlanReader.Parse(plan, _limits);
        }
        catch (PlanRejectedException e)
        {
            Rejections.Enqueue(e.Reason + ": " + e.Message);
            return HostOutcome.Recoverable((int)e.Reason);
        }

        var rows = parsed.Sequences
            .Select(s => new ForwardRow(s.TokenStart, s.TokenCount, s.PositionOffset, s.Blocks, s.RecurrentSlot, s.Sample))
            .ToList();
        var logits = model.Forward(parsed.TokenIds, rows);
        var emitted = new int[logits.Length];
        int next = 0;
        foreach (var s in parsed.Sequences.Where(s => s.Sample))
        {
            int token = TinyCpuModel.Argmax(logits[next]);
            emitted[next++] = token;
            Emitted.GetOrAdd(s.Handle, _ => []).Add(token);
        }

        return HostOutcome.Ok(StepResultWriter.Write(result, parsed, emitted, default));
    }

    public void Dispose() => _model?.Dispose();
}
