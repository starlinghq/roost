using System.Net;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>
/// R-051: <c>response_format</c> (chat completions) and <c>text.format</c> (Responses) are read into the chat client's response
/// format; a malformed format is a 400 naming the field, regular expression and grammar formats are refused by type, and a
/// well-formed JSON format reaches the client, which refuses it naming <c>response_format</c> (<c>text.format</c> on the Responses route) (no grammar engine is configured
/// here, and the worker takes no token masks yet).
/// </summary>
public sealed class ResponseFormatTests : IAsyncLifetime
{
    private const string User = """{"role":"user","content":"Hi"}""";

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
    [InlineData("""{"type":"json_object"}""", "response_format")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"},"strict":true}}""", "response_format")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"answer","description":"d","schema":{"type":"object"},"strict":null}}""", "response_format")]
    [InlineData("""{"type":"regex","regex":"a+"}""", "response_format.type")]
    [InlineData("""{"type":"grammar","grammar":"start: \"a\""}""", "response_format.type")]
    [InlineData("""{"type":"structural_tag"}""", "response_format.type")]
    [InlineData("""{}""", "response_format.type")]
    [InlineData("""[]""", "response_format")]
    [InlineData("""{"type":"json_object","schema":{}}""", "response_format.schema")]
    [InlineData("""{"type":"json_schema"}""", "response_format.json_schema")]
    [InlineData("""{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}""", "response_format.json_schema.name")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"has space","schema":{}}}""", "response_format.json_schema.name")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"a"}}""", "response_format.json_schema.schema")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"a","schema":true}}""", "response_format.json_schema.schema")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"a","schema":{},"strict":"yes"}}""", "response_format.json_schema.strict")]
    [InlineData("""{"type":"json_schema","json_schema":{"name":"a","schema":{},"x":1}}""", "response_format.json_schema.x")]
    public async Task ChatCompletionsResponseFormat(string format, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/chat/completions",
            $$"""{"model":"roost-tiny-hybrid-test","messages":[{{User}}],"response_format":{{format}}}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
        Assert.Empty(Rig.Executor.Steps);
    }

    [Fact]
    public async Task ATextFormatIsStillAccepted()
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/chat/completions",
            """{"model":"roost-tiny-hybrid-test","messages":[""" + User + """],"response_format":{"type":"text"}}""");

        Assert.True(response.StatusCode == HttpStatusCode.OK, body.GetRawText());
    }

    [Theory]
    [InlineData("""{"type":"json_object"}""", "text.format")]
    [InlineData("""{"type":"json_schema","name":"answer","schema":{"type":"object"},"strict":true}""", "text.format")]
    [InlineData("""{"type":"json_schema","schema":{"type":"object"}}""", "text.format.name")]
    [InlineData("""{"type":"regex"}""", "text.format.type")]
    public async Task ResponsesTextFormat(string format, string param)
    {
        var (response, body) = await PostAsync(Rig.Http, "/v1/responses",
            """{"model":"roost-tiny-hybrid-test","input":"Hi","text":{"format":""" + format + "}}");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
        Assert.Empty(Rig.Executor.Steps);
    }
}
