using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>Loopback by default, non-loopback only when allowed explicitly, and the <c>roost serve</c> command line.</summary>
public sealed class ListenAndCliTests
{
    private const string Sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    [Fact]
    public void TheDefaultIsLoopbackOnPort8000()
    {
        var options = new RoostServerOptions();

        Assert.Equal(IPAddress.Loopback, options.ResolveAddress());
        Assert.Equal(8000, options.Port);
        Assert.False(options.AllowNonLoopback);
        Assert.False(options.EnableSleepMode);
    }

    [Theory]
    [InlineData("127.0.0.1")]
    [InlineData("localhost")]
    [InlineData("::1")]
    [InlineData("127.0.0.2")]
    public void LoopbackHostsAreAccepted(string host) =>
        Assert.True(IPAddress.IsLoopback(new RoostServerOptions { Host = host }.ResolveAddress()));

    [Theory]
    [InlineData("0.0.0.0")]
    [InlineData("::")]
    [InlineData("192.168.1.10")]
    public void ANonLoopbackHostNeedsTheExplicitOption(string host)
    {
        var e = Assert.Throws<ArgumentException>(() => new RoostServerOptions { Host = host }.ResolveAddress());
        Assert.Contains("--allow-non-loopback", e.Message, StringComparison.Ordinal);

        Assert.Equal(IPAddress.Parse(host), new RoostServerOptions { Host = host, AllowNonLoopback = true }.ResolveAddress());
    }

    [Theory]
    [InlineData("example.com", 8000)]
    [InlineData("127.0.0.1", 70000)]
    [InlineData("127.0.0.1", -1)]
    public void ABadHostOrPortIsRefused(string host, int port) =>
        Assert.Throws<ArgumentException>(() => new RoostServerOptions { Host = host, Port = port }.ResolveAddress());

    [Fact]
    public async Task CreateRefusesANonLoopbackHost()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        Assert.Throws<ArgumentException>(() => RoostServer.Create(rig.Client, new RoostServerOptions { Host = "0.0.0.0" }));
    }

    [Fact]
    public async Task KestrelListensOnLoopbackOnlyByDefault()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });
        var app = RoostServer.Create(rig.Client, new RoostServerOptions { Port = 0 });
        await using (app)
        {
            await app.StartAsync(Ct);
            try
            {
                var address = Assert.Single(app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses);
                var uri = new Uri(address);
                Assert.Equal("127.0.0.1", uri.Host);

                using var http = new HttpClient { BaseAddress = uri };
                var response = await http.GetAsync("/v1/models", Ct);
                Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            }
            finally
            {
                await app.StopAsync(CancellationToken.None);
            }
        }
    }

    [Fact]
    public void ServeParsesItsFlags()
    {
        var c = RoostCli.Parse(["serve", "--model", "qwen3.5-0.8b", "--weights", "w.safetensors", "--weights-sha256", Sha, "--tokenizer", "t.json", "--port", "9000", "--allow-untested-sm"]);

        Assert.Equal("qwen3.5-0.8b", c.Model);
        Assert.Equal("w.safetensors", c.Weights);
        Assert.Equal(Sha, c.WeightsSha256);
        Assert.Equal("t.json", c.Tokenizer);
        Assert.True(c.AllowUntestedSm);
        Assert.Null(c.KernelDirectory);
        Assert.Equal(RoostServerOptions.DefaultHost, c.Listen.Host);
        Assert.Equal(9000, c.Listen.Port);
        Assert.False(c.Listen.AllowNonLoopback);
        Assert.False(c.Listen.EnableSleepMode);
    }

    [Fact]
    public void ServeTakesEnableSleepMode()
    {
        var c = RoostCli.Parse(["serve", "--model", "qwen3.5-0.8b", "--weights", "w.safetensors", "--weights-sha256", Sha, "--tokenizer", "t.json", "--enable-sleep-mode"]);

        Assert.True(c.Listen.EnableSleepMode);
        Assert.Contains("--enable-sleep-mode", RoostCli.Usage, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("--gpu-memory-utilization", "unknown option '--gpu-memory-utilization'")]
    [InlineData("--port", "--port needs a value")]
    public void ServeRefusesAnUnknownOrIncompleteFlagByName(string flag, string message)
    {
        var e = Assert.Throws<ArgumentException>(() => RoostCli.Parse(["serve", "--model", "qwen3.5-0.8b", flag]));
        Assert.Contains(message, e.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(new[] { "run" }, "unknown command 'run'")]
    [InlineData(new[] { "serve", "--model", "llama", "--weights", "w", "--weights-sha256", Sha, "--tokenizer", "t" }, "unknown model profile 'llama'")]
    [InlineData(new[] { "serve", "--model", "qwen3.5-0.8b", "--weights", "w", "--weights-sha256", "abc", "--tokenizer", "t" }, "--weights-sha256 must be 64 hex digits")]
    [InlineData(new[] { "serve", "--model", "qwen3.5-0.8b", "--weights-sha256", Sha, "--tokenizer", "t" }, "--weights is required")]
    public void ServeRefusesBadArguments(string[] args, string message)
    {
        var e = Assert.Throws<ArgumentException>(() => RoostCli.Parse(args));
        Assert.Contains(message, e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ServeRefusesANonLoopbackHostBeforeStartingTheWorker()
    {
        using var error = new StringWriter();

        var code = await RoostCli.RunAsync(["serve", "--model", "qwen3.5-0.8b", "--weights", "missing.safetensors", "--weights-sha256", Sha, "--tokenizer", "missing.json", "--host", "0.0.0.0"], error);

        Assert.Equal(2, code);
        Assert.Contains("--allow-non-loopback", error.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task KestrelsRefusalOfABodyIsAClientError()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });
        var app = RoostServer.Create(rig.Client, new RoostServerOptions { Port = 0 }, web => web.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = 16));
        await using (app)
        {
            await app.StartAsync(Ct);
            try
            {
                var address = Assert.Single(app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses);
                using var http = new HttpClient { BaseAddress = new Uri(address) };

                var (response, body) = await PostAsync(http, "/v1/chat/completions", "{\"padding\":\"" + new string('x', 256) + "\"}");

                Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
                AssertErrorBody(body, "bad_request", null);
            }
            finally
            {
                await app.StopAsync(CancellationToken.None);
            }
        }
    }

    [Fact]
    public void ServeParsesAllowedHosts()
    {
        var c = RoostCli.Parse(["serve", "--model", "qwen3.5-0.8b", "--weights", "w", "--weights-sha256", Sha, "--tokenizer", "t", "--allowed-host", "a.example, b.example"]);

        Assert.Equal(["a.example", "b.example"], c.Listen.AllowedHosts);
    }

    [Fact]
    public async Task ServeReportsACorruptTokenizer()
    {
        var path = Path.Combine(Path.GetTempPath(), "roost-corrupt-" + Guid.NewGuid().ToString("N") + ".json");
        await File.WriteAllTextAsync(path, "{ not json", Ct);
        try
        {
            using var error = new StringWriter();

            var code = await RoostCli.RunAsync(["serve", "--model", "qwen3.5-0.8b", "--weights", "missing.safetensors", "--weights-sha256", Sha, "--tokenizer", path], error);

            Assert.Equal(1, code);
            Assert.Contains("the worker did not start", error.ToString(), StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task ServeReportsAWorkerThatCannotStart()
    {
        using var error = new StringWriter();

        var code = await RoostCli.RunAsync(["serve", "--model", "qwen3.5-0.8b", "--weights", "missing.safetensors", "--weights-sha256", Sha, "--tokenizer", "missing.json"], error);

        Assert.Equal(1, code);
        Assert.Contains("the worker did not start", error.ToString(), StringComparison.Ordinal);
    }
}
