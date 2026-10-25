using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using Starlings.Roost.Abi;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost.Tests.Support;

/// <summary>What a <see cref="ScriptedExecutor"/> does; encodable on a command line for the process transport.</summary>
internal sealed record Script
{
    /// <summary>Tokens every sequence emits, in order; after them it emits <see cref="StopId"/> (or repeats the last one when <see cref="NeverStop"/>).</summary>
    public required int[] Tokens { get; init; }

    public bool NeverStop { get; init; }

    /// <summary>1-based step number that faults, 0 for none.</summary>
    public int FaultAtStep { get; init; }

    public FaultClass Fault { get; init; } = FaultClass.None;

    /// <summary>Sleep per step, to give a test time to cancel.</summary>
    public int DelayMs { get; init; }

    public string Encode() => string.Create(CultureInfo.InvariantCulture,
        $"{string.Join('.', Tokens)};{(NeverStop ? 1 : 0)};{FaultAtStep};{(uint)Fault};{DelayMs}");

    public static Script Decode(string text)
    {
        var p = text.Split(';');
        return new Script
        {
            Tokens = p[0].Length == 0 ? [] : [.. p[0].Split('.').Select(s => int.Parse(s, CultureInfo.InvariantCulture))],
            NeverStop = p[1] == "1",
            FaultAtStep = int.Parse(p[2], CultureInfo.InvariantCulture),
            Fault = (FaultClass)uint.Parse(p[3], CultureInfo.InvariantCulture),
            DelayMs = int.Parse(p[4], CultureInfo.InvariantCulture),
        };
    }
}

/// <summary>One step as the worker saw it.</summary>
internal sealed record SeenStep(ulong StepId, int Sequences, int Tokens, int[] ChunkTokens, int[] PositionOffsets, uint[] MaxNewTokensLeft, int[] StopIds);

/// <summary>
/// A step executor that runs no model: it reads each plan with the eager worker's own <see cref="StepPlanReader"/> against the
/// eager worker's <see cref="PlanLimits"/> for the test profile, emits the script's next token for every sampling sequence,
/// and writes the result with the eager worker's <see cref="StepResultWriter"/> (which sets the stop-id and length finish
/// flags). Everything between the chat client and the forward pass is therefore the GPU path's code.
/// </summary>
internal sealed class ScriptedExecutor : IStepExecutor
{
    public static readonly ImmutableArray<byte> ManifestDigest = [.. SHA256.HashData(Encoding.UTF8.GetBytes("roost-chat-test-manifest"))];
    public static readonly ImmutableArray<byte> WeightsDigest = [.. SHA256.HashData(Encoding.UTF8.GetBytes("roost-chat-test-weights"))];

    private readonly Script _script;
    private readonly PlanLimits _limits;
    private readonly int _stopId;
    private readonly Dictionary<ulong, int> _emitted = [];
    private int _steps;

    public ScriptedExecutor(Script script, PlanLimits limits, int stopId)
    {
        _script = script;
        _limits = limits;
        _stopId = stopId;
    }

    /// <summary>Every step seen, in order (in-process only).</summary>
    public ConcurrentQueue<SeenStep> Steps { get; } = new();

    /// <summary>Plans the eager reader refused (must stay empty).</summary>
    public ConcurrentQueue<string> Rejections { get; } = new();

    public WorkerIdentity Identity => new(KernelAbi.Version, ManifestDigest, WeightsDigest);

    public HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid)
    {
        deviceUuid.Fill(0xC5);
        return deviceOrdinal == 0 ? HandshakeReject.None : HandshakeReject.Device;
    }

    public void RegisterSegment(nint address, long bytes)
    {
    }

    public HandshakeReject LoadWeights() => HandshakeReject.None;

    public StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result)
    {
        _steps++;
        ParsedPlan parsed;
        try
        {
            parsed = StepPlanReader.Parse(plan, _limits);
        }
        catch (PlanRejectedException e)
        {
            Rejections.Enqueue(e.Reason + ": " + e.Message);
            return StepOutcome.Recoverable((int)e.Reason);
        }

        Steps.Enqueue(new SeenStep(
            parsed.StepId,
            parsed.Sequences.Length,
            parsed.TokenIds.Length,
            [.. parsed.Sequences.Select(static s => s.TokenCount)],
            [.. parsed.Sequences.Select(static s => s.PositionOffset)],
            [.. parsed.Sequences.Select(static s => s.MaxNewTokensLeft)],
            parsed.StopIds));

        if (_script.DelayMs > 0)
        {
            Thread.Sleep(_script.DelayMs);
        }

        if (_steps == _script.FaultAtStep)
        {
            return _script.Fault == FaultClass.Sticky ? StepOutcome.Sticky(700) : StepOutcome.Recoverable(11);
        }

        var tokens = new List<int>();
        foreach (var s in parsed.Sequences)
        {
            if (!s.Sample)
            {
                continue;
            }

            if (s.PositionOffset == 0 || !_emitted.ContainsKey(s.Handle))
            {
                _emitted[s.Handle] = 0;
            }

            var n = _emitted[s.Handle]++;
            tokens.Add(n < _script.Tokens.Length ? _script.Tokens[n] : _script.NeverStop ? _script.Tokens[^1] : _stopId);
        }

        int bytes = StepResultWriter.Write(result, parsed, [.. tokens], default);
        return StepOutcome.Ok(bytes);
    }

    public void Dispose()
    {
    }
}
