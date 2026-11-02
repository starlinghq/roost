using System.Net;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;
using static Starlings.Roost.Server.Tests.Support.Prometheus;

namespace Starlings.Roost.Server.Tests;

/// <summary>Cold-start stage timing (R-046): stages measured on an injected clock, logged as they end, and exported on <c>GET /metrics</c>.</summary>
public sealed class ColdStartTests
{
    [Fact]
    public void StagesAreMeasuredInOrderAndLogged()
    {
        var clock = new SteppedTime();
        using var log = new StringWriter();
        var coldStart = new ColdStart(clock, log);

        coldStart.Record(ColdStart.Runtime, TimeSpan.FromMilliseconds(40));
        int profile = coldStart.Measure(ColdStart.Profile, () =>
        {
            clock.Advance(TimeSpan.FromMilliseconds(250));
            return 7;
        });

        Assert.Equal(7, profile);
        Assert.Equal([new ColdStartStage(ColdStart.Runtime, TimeSpan.FromMilliseconds(40)), new ColdStartStage(ColdStart.Profile, TimeSpan.FromMilliseconds(250))], coldStart.Stages);
        Assert.Equal(TimeSpan.FromMilliseconds(290), coldStart.Total);
        coldStart.WriteTotal();
        Assert.Equal(
            ["roost: cold start: runtime 0.040 s", "roost: cold start: profile 0.250 s", "roost: cold start: total 0.290 s (runtime, profile)"],
            log.ToString().TrimEnd().Split(Environment.NewLine));
    }

    [Fact]
    public async Task AFailedStageIsStillRecorded()
    {
        var clock = new SteppedTime();
        var coldStart = new ColdStart(clock, null);

        await Assert.ThrowsAsync<IOException>(() => coldStart.MeasureAsync<int>(ColdStart.Worker, async () =>
        {
            await Task.Yield();
            clock.Advance(TimeSpan.FromSeconds(3));
            throw new IOException("the worker did not start");
        }));

        Assert.Equal(new ColdStartStage(ColdStart.Worker, TimeSpan.FromSeconds(3)), Assert.Single(coldStart.Stages));
    }

    [Fact]
    public void TheProcessStartTimeIsReadable() =>
        Assert.True(ColdStart.SinceProcessStart() is null or { Ticks: >= 0 });

    [Fact]
    public async Task TheStagesAreExportedOnMetrics()
    {
        var coldStart = new ColdStart(new SteppedTime(), null);
        coldStart.Record(ColdStart.Arguments, TimeSpan.FromMilliseconds(2));
        coldStart.Record(ColdStart.Worker, TimeSpan.FromSeconds(4.5));
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") }, new RoostServerOptions { ColdStart = coldStart });

        var response = await rig.Http.GetAsync("/metrics", Ct);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var (samples, types) = Parse(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal("gauge", types["roost_cold_start_stage_seconds"]);
        Assert.Equal("gauge", types["roost_cold_start_seconds"]);
        Assert.Equal(0.002, Value(samples, "roost_cold_start_stage_seconds", ("stage", ColdStart.Arguments), ("model_name", ServerRig.ModelId)));
        Assert.Equal(4.5, Value(samples, "roost_cold_start_stage_seconds", ("stage", ColdStart.Worker)));
        Assert.Equal(4.502, Value(samples, "roost_cold_start_seconds"), 9);
    }

    [Fact]
    public async Task WithoutAColdStartNoFamilyIsExported()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var text = await rig.Http.GetStringAsync("/metrics", Ct);

        Assert.DoesNotContain("roost_cold_start", text, StringComparison.Ordinal);
    }

    /// <summary>A clock whose timestamps move only when told to.</summary>
    private sealed class SteppedTime : TimeProvider
    {
        private long _ticks;

        public override long TimestampFrequency => TimeSpan.TicksPerSecond;

        public override long GetTimestamp() => Interlocked.Read(ref _ticks);

        public void Advance(TimeSpan by) => Interlocked.Add(ref _ticks, by.Ticks);
    }
}
