using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.TestHost;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Models.Tiny;
using Starlings.Roost.Server.Tests.Support;

namespace Starlings.Roost.Server.Tests;

/// <summary>
/// The sleep-mode routes (R-100): off by default, loopback clients only, and over a worker that runs roost-tiny-hybrid on
/// <c>CpuOps</c> they sleep, wake (with the wake time in the body), report the state and replace the weights from a verified file.
/// </summary>
public sealed class SleepModeRouteTests : IDisposable
{
    private static readonly RoostServerOptions SleepMode = new() { EnableSleepMode = true };

    private readonly DirectoryInfo _dir = Directory.CreateTempSubdirectory("roost-r100-server-");

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _dir.Delete(recursive: true);

    [Fact]
    public async Task TheRoutesAreNotMappedByDefault()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });
        foreach (var (method, path) in new[] { ("POST", "/sleep"), ("POST", "/wake_up"), ("GET", "/is_sleeping"), ("POST", "/update_weights") })
        {
            var (status, body) = await SendAsync(rig, method, path, null, IPAddress.Loopback);
            Assert.Equal(404, status);
            Http.AssertErrorBody(body, "not_found", null);
        }
    }

    [Fact]
    public async Task SleepAndWakeOverHttpWithTheWakeTimeInTheBody()
    {
        var path = Path.Combine(_dir.FullName, "model.safetensors");
        var locked = TinyCheckpoint.Write(path, TinyConfig.Hybrid.Seed);
        await using var rig = await ServerRig.StartTinyAsync(path, locked, SleepMode);

        var (status, body) = await SendAsync(rig, "POST", "/sleep?level=2", null, IPAddress.Loopback);
        Assert.Equal(200, status);
        Assert.Equal("sleep", body.GetProperty("operation").GetString());
        Assert.True(body.GetProperty("is_sleeping").GetBoolean());
        Assert.True(body.GetProperty("bytes").GetInt64() > 0);
        Assert.Equal(JsonValueKind.Null, body.GetProperty("weights_sha256").ValueKind);

        (status, body) = await SendAsync(rig, "GET", "/is_sleeping", null, IPAddress.Loopback);
        Assert.Equal(200, status);
        Assert.True(body.GetProperty("is_sleeping").GetBoolean());

        (status, body) = await SendAsync(rig, "POST", "/v1/chat/completions", """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}]}""", IPAddress.Loopback);
        Assert.Equal(503, status);
        Http.AssertErrorBody(body, "insufficient_device_memory", null);

        (status, body) = await SendAsync(rig, "POST", "/wake_up", null, IPAddress.Loopback);
        Assert.Equal(200, status);
        Assert.Equal("wake_up", body.GetProperty("operation").GetString());
        Assert.False(body.GetProperty("is_sleeping").GetBoolean());
        Assert.True(body.GetProperty("duration_ms").GetDouble() > 0);
        Assert.Equal(locked.Sha256, body.GetProperty("weights_sha256").GetString());
        Assert.NotNull(rig.Client.LastWakeDuration);

        (status, body) = await SendAsync(rig, "POST", "/wake_up", null, IPAddress.Loopback);
        Assert.Equal(400, status);
        Http.AssertErrorBody(body, "invalid_option", null);

        (status, _) = await SendAsync(rig, "POST", "/v1/chat/completions", """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}],"max_tokens":4}""", IPAddress.Loopback);
        Assert.Equal(200, status);
    }

    [Fact]
    public async Task UpdateWeightsChecksTheBodyAndTheFile()
    {
        var path = Path.Combine(_dir.FullName, "model.safetensors");
        var locked = TinyCheckpoint.Write(path, TinyConfig.Hybrid.Seed);
        var nextPath = Path.Combine(_dir.FullName, "next.safetensors");
        var next = TinyCheckpoint.Write(nextPath, TinyConfig.Hybrid.Seed + 1);
        await using var rig = await ServerRig.StartTinyAsync(path, locked, SleepMode);

        var (status, body) = await SendAsync(rig, "POST", "/update_weights", """{"path":"relative.safetensors","size":1,"sha256":"00"}""", IPAddress.Loopback);
        Assert.Equal(400, status);
        Http.AssertErrorBody(body, "invalid_option", "path");

        (status, body) = await SendAsync(rig, "POST", "/update_weights", Body(nextPath, next.Size, next.Sha256) + " ", IPAddress.Loopback);
        Assert.Equal(200, status);
        Assert.Equal("update_weights", body.GetProperty("operation").GetString());
        Assert.Equal(next.Sha256, body.GetProperty("weights_sha256").GetString());

        // The lock the caller gives is the original file's: the bytes read do not match it.
        (status, body) = await SendAsync(rig, "POST", "/update_weights", Body(nextPath, locked.Size, locked.Sha256), IPAddress.Loopback);
        Assert.Equal(500, status);
        Http.AssertErrorBody(body, "asset_mismatch", null);

        (status, body) = await SendAsync(rig, "POST", "/update_weights", """{"path":"/x","size":1,"sha256":"aa","extra":1}""", IPAddress.Loopback);
        Assert.Equal(400, status);
        Http.AssertErrorBody(body, "invalid_option", "extra");

        (status, body) = await SendAsync(rig, "POST", "/sleep?level=3", null, IPAddress.Loopback);
        Assert.Equal(400, status);
        Http.AssertErrorBody(body, "invalid_option", "level");
    }

    [Theory]
    [InlineData("192.0.2.10")]
    [InlineData("2001:db8::1")]
    [InlineData(null)]
    public async Task OnlyLoopbackClientsMayUseThem(string? remote)
    {
        var path = Path.Combine(_dir.FullName, "model.safetensors");
        var locked = TinyCheckpoint.Write(path, TinyConfig.Hybrid.Seed);
        await using var rig = await ServerRig.StartTinyAsync(path, locked, SleepMode);
        var address = remote is null ? null : IPAddress.Parse(remote);
        foreach (var (method, route) in new[] { ("POST", "/sleep"), ("POST", "/wake_up"), ("GET", "/is_sleeping"), ("POST", "/update_weights") })
        {
            var (status, body) = await SendAsync(rig, method, route, Body(path, locked.Size, locked.Sha256), address);
            Assert.Equal(403, status);
            Http.AssertErrorBody(body, "forbidden", null);
        }

        Assert.False(rig.Client.IsSleeping);
        var (ok, _) = await SendAsync(rig, "GET", "/is_sleeping", null, IPAddress.IPv6Loopback);
        Assert.Equal(200, ok);
    }

    [Fact]
    public async Task AWorkerWithoutTheLifecycleAnswersProfileUnsupported()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") }, SleepMode);
        var (status, body) = await SendAsync(rig, "POST", "/sleep", null, IPAddress.Loopback);
        Assert.Equal(400, status);
        Http.AssertErrorBody(body, "profile_unsupported", null);
    }

    [Fact]
    public async Task AWeightsPathLongerThanAStepPlanDoesNotCostTheWorker()
    {
        var path = Path.Combine(_dir.FullName, "model.safetensors");
        var locked = TinyCheckpoint.Write(path, TinyConfig.Hybrid.Seed);
        await using var rig = await ServerRig.StartTinyAsync(path, locked, SleepMode);

        // Within the 4096-byte limit, longer than this profile's step plan: the control message must still fit the buffer.
        var (status, body) = await SendAsync(rig, "POST", "/update_weights", Body("/" + new string('a', 4000), locked.Size, locked.Sha256), IPAddress.Loopback);
        Assert.NotEqual(200, status);
        Assert.DoesNotContain("worker_restarted", body.GetRawText(), StringComparison.Ordinal);

        (status, _) = await SendAsync(rig, "POST", "/update_weights", Body(path, locked.Size, locked.Sha256), IPAddress.Loopback);
        Assert.Equal(200, status);
    }

    private static string Body(string path, long size, string sha) =>
        JsonSerializer.Serialize(new Dictionary<string, object> { ["path"] = path, ["size"] = size, ["sha256"] = sha });

    private static async Task<(int Status, JsonElement Body)> SendAsync(ServerRig rig, string method, string pathAndQuery, string? json, IPAddress? remote)
    {
        var server = rig.App.GetTestServer();
        var context = await server.SendAsync(c =>
        {
            c.Request.Method = method;
            var q = pathAndQuery.IndexOf('?', StringComparison.Ordinal);
            c.Request.Path = q < 0 ? pathAndQuery : pathAndQuery[..q];
            c.Request.QueryString = q < 0 ? QueryString.Empty : new QueryString(pathAndQuery[q..]);
            c.Connection.RemoteIpAddress = remote;
            if (json is not null)
            {
                c.Request.ContentType = "application/json";
                c.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(json));
            }
        }, Ct);

        using var reader = new StreamReader(context.Response.Body);
        var text = await reader.ReadToEndAsync(Ct);
        using var doc = JsonDocument.Parse(text);
        return (context.Response.StatusCode, doc.RootElement.Clone());
    }
}
