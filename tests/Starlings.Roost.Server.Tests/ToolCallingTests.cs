using System.Net;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary>
/// Tool calls and reasoning on <c>POST /v1/chat/completions</c> (R-053): <c>tools</c>, <c>tool_choice</c>, assistant
/// <c>tool_calls</c> and <c>tool</c> messages in, OpenAI's <c>tool_calls</c> and vLLM's <c>reasoning_content</c> out, whole and
/// streamed. The scripted completions use the Qwen3.5 template's format.
/// </summary>
public sealed class ToolCallingTests
{
    private const int Blocks = 128; // a 2,048-token context: the tools prompt is about 1,500 byte tokens

    private const string WeatherCall = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

    private const string Tools = """
        "tools":[{"type":"function","function":{"name":"get_weather","description":"Get the weather.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]},"strict":false}}]
        """;

    private const string Ask = """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Weather in Paris?"}],""";

    private static RoostToolDefinition Weather => new(
        "get_weather", "Get the weather.", """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");

    private static Task<ServerRig> StartAsync(string completion) =>
        ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(completion) }, attentionBlocks: Blocks);

    [Fact]
    public async Task AWholeResponseCarriesToolCallsWithNullContentAndFinishReasonToolCalls()
    {
        await using var rig = await StartAsync(WeatherCall);

        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions", Ask + Tools + ""","tool_choice":"auto","parallel_tool_calls":true}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var choice = body.GetProperty("choices")[0];
        Assert.Equal("tool_calls", choice.GetProperty("finish_reason").GetString());
        var message = choice.GetProperty("message");
        Assert.Equal(JsonValueKind.Null, message.GetProperty("content").ValueKind);
        Assert.False(message.TryGetProperty("reasoning_content", out _));
        var call = Assert.Single(message.GetProperty("tool_calls").EnumerateArray());
        Assert.StartsWith("call_", call.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.Equal("function", call.GetProperty("type").GetString());
        Assert.Equal("get_weather", call.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("""{"city":"Paris"}""", call.GetProperty("function").GetProperty("arguments").GetString());

        var prompt = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Weather in Paris?")], [Weather]);
        Assert.Equal(ServerRig.Encode(prompt).Length, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
    }

    [Fact]
    public async Task AStreamSendsReasoningDeltasThenOneToolCallDeltaThenFinishReasonToolCalls()
    {
        await using var rig = await StartAsync("Look it up.\n</think>\n\n" + WeatherCall);

        var http = await rig.Http.PostAsync("/v1/chat/completions", Json(Ask + Tools + ""","stream":true,"chat_template_kwargs":{"enable_thinking":true}}"""), Ct);
        var events = Events(await http.Content.ReadAsStringAsync(Ct));

        Assert.Equal("[DONE]", events[^1]);
        var deltas = events[..^1].Select(e => JsonDocument.Parse(e).RootElement.GetProperty("choices")[0]).ToList();
        var reasoning = string.Concat(deltas.Select(d => d.GetProperty("delta").TryGetProperty("reasoning_content", out var r) ? r.GetString() : string.Empty));
        Assert.Equal("Look it up.", reasoning);
        Assert.DoesNotContain(deltas, d => d.GetProperty("delta").TryGetProperty("content", out var c) && c.GetString()!.Length > 0);
        var callDeltas = deltas.Where(d => d.GetProperty("delta").TryGetProperty("tool_calls", out _)).ToList();
        var call = Assert.Single(Assert.Single(callDeltas).GetProperty("delta").GetProperty("tool_calls").EnumerateArray());
        Assert.Equal(0, call.GetProperty("index").GetInt32());
        Assert.Equal("get_weather", call.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("""{"city":"Paris"}""", call.GetProperty("function").GetProperty("arguments").GetString());
        Assert.Equal("tool_calls", deltas[^1].GetProperty("finish_reason").GetString());

        var prompt = Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User("Weather in Paris?")], [Weather], enableThinking: true);
        Assert.Equal(ServerRig.Encode(prompt).Length, rig.Executor.Steps.Sum(s => s.Tokens) - ServerRig.Encode("Look it up.\n</think>\n\n" + WeatherCall).Length);
    }

    [Fact]
    public async Task AWholeResponseCarriesReasoningContent()
    {
        await using var rig = await StartAsync("Greeting.\n</think>\n\nHello!");

        var (_, body) = await PostAsync(rig.Http, "/v1/chat/completions",
            """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}],"chat_template_kwargs":{"enable_thinking":true}}""");

        var message = body.GetProperty("choices")[0].GetProperty("message");
        Assert.Equal("Hello!", message.GetProperty("content").GetString());
        Assert.Equal("Greeting.", message.GetProperty("reasoning_content").GetString());
        Assert.False(message.TryGetProperty("tool_calls", out _));
        Assert.Equal("stop", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
    }

    [Fact]
    public async Task AToolCallAndItsResultInTheConversationAreRendered()
    {
        await using var rig = await StartAsync("Sunny.");

        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions", """
            {"model":"roost-tiny-hybrid-test",
             "messages":[{"role":"user","content":"Weather in Paris?"},
                         {"role":"assistant","content":null,"reasoning_content":"Look it up.","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"}}]},
                         {"role":"tool","tool_call_id":"call_1","content":"18 C"}],
            """ + Tools + "}");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("Sunny.", body.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString());
        var prompt = Qwen35RoostChatTemplate.Instance.Render(
            [
                RoostChatMessage.User("Weather in Paris?"),
                RoostChatMessage.Assistant(string.Empty, "Look it up.", [new RoostToolCall("get_weather", """{"city":"Paris"}""")]),
                RoostChatMessage.Tool("18 C"),
            ],
            [Weather]);
        Assert.Equal(ServerRig.Encode(prompt).Length, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
    }

    [Fact]
    public async Task ToolChoiceNoneReturnsTheMarkupAsContent()
    {
        await using var rig = await StartAsync(WeatherCall);

        var (_, body) = await PostAsync(rig.Http, "/v1/chat/completions", Ask + Tools + ""","tool_choice":"none"}""");

        var choice = body.GetProperty("choices")[0];
        Assert.Equal(WeatherCall, choice.GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
    }
}
