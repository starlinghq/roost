using System.Collections.Immutable;
using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Responses;
using Starlings.Roost.Server.Routing;
using Starlings.Roost.Server.Tests.Support;
using Starlings.Roost.WorkerHost;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>The checks that keep a local browser and over-large or unreadable bodies from reaching the routes (#157).</summary>
public sealed class RequestHardeningTests : IAsyncLifetime
{
    private const string Model = "\"model\":\"" + ServerRig.ModelId + "\"";
    private const string Chat = "{" + Model + ",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";

    private ServerRig? _rig;

    private ServerRig Rig => _rig!;

    public async ValueTask InitializeAsync() => _rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

    public async ValueTask DisposeAsync()
    {
        if (_rig is not null)
        {
            await _rig.DisposeAsync();
        }
    }

    [Theory]
    [InlineData("localhost:8000", null, true)]
    [InlineData("LOCALHOST", null, true)]
    [InlineData("127.0.0.1:8000", null, true)]
    [InlineData("[::1]:8000", null, true)]
    [InlineData("127.0.0.1:8000", "http://localhost:3000", true)]
    [InlineData("127.0.0.1:8000", "http://127.0.0.1:3000", true)]
    [InlineData("127.0.0.1:8000", "http://[::1]:3000", true)]
    [InlineData("127.0.0.1:8000", "", true)]
    // DNS rebinding: the page's own name reaches a loopback listener.
    [InlineData("evil.example:8000", null, false)]
    [InlineData("evil.example:8000", "http://evil.example:8000", false)]
    // A cross-origin page, and the opaque origin a sandboxed page sends.
    [InlineData("127.0.0.1:8000", "https://evil.example", false)]
    [InlineData("127.0.0.1:8000", "null", false)]
    public void ALoopbackListenerAcceptsThisMachineOnly(string host, string? origin, bool accepted) =>
        Assert.Equal(accepted, RequestOriginGuard.Refusal(new HostString(host), origin, loopbackListener: true) is null);

    [Theory]
    [InlineData("roost.internal:8000", null, true)]
    [InlineData("roost.internal:8000", "http://roost.internal:3000", true)]
    [InlineData("roost.internal:8000", "https://evil.example", false)]
    public void AnOpenListenerAcceptsAnyHostAndStillRefusesOtherOrigins(string host, string? origin, bool accepted) =>
        Assert.Equal(accepted, RequestOriginGuard.Refusal(new HostString(host), origin, loopbackListener: false) is null);

    [Fact]
    public async Task APageOnAnotherOriginIsRefusedBeforeAnyRouteRuns()
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions") { Content = Json(Chat) };
        request.Headers.Add("Origin", "https://evil.example");

        using var response = await Rig.Http.SendAsync(request, Ct);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        using var body = JsonDocument.Parse(await response.Content.ReadAsStringAsync(Ct));
        AssertErrorBody(body.RootElement, "forbidden_origin", null);
    }

    [Fact]
    public async Task ARebindingHostIsRefused()
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, "/metrics");
        request.Headers.Host = "evil.example";

        using var response = await Rig.Http.SendAsync(request, Ct);

        Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
    }

    [Fact]
    public async Task AClientThatSendsNoOriginIsUnaffected()
    {
        var (response, _) = await PostAsync(Rig.Http, "/v1/chat/completions", Chat);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Theory]
    [InlineData("{\"model\":\"\\ud800\"}")]
    [InlineData("{\"\\ud800\":1}")]
    [InlineData("{\"messages\":[{\"role\":\"user\",\"content\":\"\\udc00\"}]}")]
    public async Task TextThatIsNotUnicodeIsA400(string json)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/chat/completions", json);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", null);
    }

    [Fact]
    public async Task InvalidUtf8IsA400()
    {
        var bytes = new ByteArrayContent([.. Encoding.UTF8.GetBytes("{\"model\":\""), 0xFF, .. Encoding.UTF8.GetBytes("\"}")]);
        using var response = await Rig.Http.PostAsync("/v1/chat/completions", bytes, Ct);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Theory]
    [MemberData(nameof(OverLimit))]
    public async Task ResponseFieldsThatAreKeptHaveLimits(string field, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/responses", "{" + Model + ",\"input\":\"Hi\"," + field + "}");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
    }

    public static TheoryData<string, string> OverLimit() => new()
    {
        { "\"metadata\":{\"k\":\"" + new string('v', 513) + "\"}", "metadata" },
        { "\"metadata\":{\"" + new string('k', 65) + "\":\"v\"}", "metadata" },
        { "\"metadata\":{" + string.Join(',', Enumerable.Range(0, 17).Select(i => $"\"k{i}\":\"v\"")) + "}", "metadata" },
        { "\"user\":\"" + new string('u', 257) + "\"", "user" },
    };

    [Fact]
    public async Task FieldsAtTheLimitsAreAccepted()
    {
        var pairs = string.Join(',', Enumerable.Range(0, 16).Select(i => $"\"k{i:D2}{new string('k', 61)}\":\"{new string('v', 512)}\""));
        var (response, _) = await PostAsync(Rig.Http, "/v1/responses", "{" + Model + ",\"input\":\"Hi\",\"metadata\":{" + pairs + "},\"user\":\"" + new string('u', 256) + "\"}");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public void AStoredResponseDropsItsPromptWhenItEnds()
    {
        Action<StoredResponse>[] endings =
        [
            s => s.Finish(null, null),
            s => s.Fail(new Starlings.Roost.Requests.RoostError(Starlings.Roost.Requests.RoostErrorCode.Internal, "x")),
            s => s.MarkCancelled(),
        ];
        foreach (var end in endings)
        {
            var messages = new[] { new Microsoft.Extensions.AI.ChatMessage(Microsoft.Extensions.AI.ChatRole.User, "kept until the end") };
            var stored = new StoredResponse(new ResponsesRequest(messages, new(), false, true, false, [new("k", "v")], "u"), "m", DateTimeOffset.UnixEpoch);
            Assert.NotEmpty(stored.Request.Messages);

            end(stored);

            Assert.Empty(stored.Request.Messages);
            Assert.Equal("u", stored.Request.User);
            Assert.Single(stored.Request.Metadata);
        }
    }

    [Fact]
    public void AnOverlongWeightsPathIsRefusedBeforeAnythingReachesAWorker()
    {
        var path = "/" + new string('a', ControlMessages.MaxPathBytes);
        var source = new WeightSource(path, 1, ImmutableArray.Create(new byte[32]));

        var e = Assert.Throws<ArgumentException>(source.Validate);

        Assert.Contains("UTF-8", e.Message, StringComparison.Ordinal);
    }
}
