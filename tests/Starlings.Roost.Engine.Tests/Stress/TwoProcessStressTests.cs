using System.Globalization;

namespace Starlings.Roost.Engine.Tests.Stress;

/// <summary>
/// A host and a worker in two processes over one shared-memory segment, with randomised delays on both sides and a
/// check of every slot and every buffer word. Length: ROOST_RING_STRESS_STEPS (default 300000);
/// ROOST_RING_STRESS_SEED fixes the seed; ROOST_RING_STRESS_LOG names a file that gets the result line. This exercises the ring code on this machine's memory model (x86-64 is
/// stronger than the .NET model; a pass here does not show the ordering argument holds on ARM64).
/// </summary>
public sealed class TwoProcessStressTests
{
    [Fact]
    public void EverySlotAndBufferArrivesOnceInOrderAndIntact()
    {
        var steps = Env("ROOST_RING_STRESS_STEPS", 300_000UL);
        var seed = Env("ROOST_RING_STRESS_SEED", (ulong)Environment.TickCount64);
        var outcome = StressHost.Run(steps, seed, TimeSpan.FromMinutes(Math.Max(2, steps / 1_000_000UL * 4)));

        Assert.Equal(steps, outcome.Steps);
        // Both rings must have hit full and empty, or the test did not exercise what it claims to.
        Assert.True(outcome.PublishFull > 0, "the plan ring never filled");
        Assert.True(outcome.ResultEmpty > 0, "the result ring never ran empty");
        Assert.Contains("OK steps=" + steps.ToString(CultureInfo.InvariantCulture), outcome.WorkerSummary, StringComparison.Ordinal);
        Assert.Contains("allocatedBytes=0", outcome.WorkerSummary, StringComparison.Ordinal);
        Assert.Equal(0, outcome.HostAllocatedBytes);
        Assert.False(outcome.WorkerSummary.Contains("planEmpty=0 ", StringComparison.Ordinal), "the worker never found the plan ring empty");
        Assert.False(outcome.WorkerSummary.Contains("twoInFlight=0 ", StringComparison.Ordinal), "two plans were never in flight");
        var line = $"seed={seed} hostPublishFull={outcome.PublishFull} hostResultEmpty={outcome.ResultEmpty} hostAllocatedBytes={outcome.HostAllocatedBytes} worker: {outcome.WorkerSummary}";
        TestContext.Current.SendDiagnosticMessage(line);
        var log = Environment.GetEnvironmentVariable("ROOST_RING_STRESS_LOG");
        if (!string.IsNullOrEmpty(log))
        {
            File.AppendAllText(log, line + Environment.NewLine);
        }
    }

    private static ulong Env(string name, ulong fallback) =>
        ulong.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.None, CultureInfo.InvariantCulture, out var v) ? v : fallback;
}
