using System.Net;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary><c>POST /v1/messages</c> and <c>POST /v1/messages/count_tokens</c> (Anthropic's shapes) over the scripted worker.</summary>
public sealed class MessagesTests
{
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";
    private const string Hi = """{"model":"roost-tiny-hybrid-test","max_tokens":64,"messages":[{"role":"user","content":"Hi"}]""";

    private static int PromptTokens(params RoostChatMessage[] messages) =>
        ServerRig.Encode(Qwen35RoostChatTemplate.Instance.Render(messages)).Length;

    [Fact]
    public async Task AWholeRequestReturnsAMessage()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (response, body) = await PostAsync(rig.Http, "/v1/messages", Hi + "}");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        Assert.Equal(["id", "type", "role", "model", "content", "stop_reason", "stop_sequence", "usage"], body.EnumerateObject().Select(p => p.Name));
        Assert.StartsWith("msg_", body.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.Equal("message", body.GetProperty("type").GetString());
        Assert.Equal("assistant", body.GetProperty("role").GetString());
        Assert.Equal(ServerRig.ModelId, body.GetProperty("model").GetString());
        var block = Assert.Single(body.GetProperty("content").EnumerateArray());
        Assert.Equal("text", block.GetProperty("type").GetString());
        Assert.Equal(Reply, block.GetProperty("text").GetString());
        Assert.Equal("end_turn", body.GetProperty("stop_reason").GetString());
        Assert.Equal(JsonValueKind.Null, body.GetProperty("stop_sequence").ValueKind);
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), body.GetProperty("usage").GetProperty("input_tokens").GetInt32());
        Assert.Equal(ServerRig.Encode(Reply).Length + 1, body.GetProperty("usage").GetProperty("output_tokens").GetInt32());
    }

    [Fact]
    public async Task SystemBlocksAndAcceptedFieldsRenderAsTheSystemMessage()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/messages", """
            {"model":"roost-tiny-hybrid-test","max_tokens":8,"temperature":0,"top_p":1,"stream":false,"tools":[],
             "tool_choice":{"type":"none"},"thinking":{"type":"disabled"},"metadata":{"user_id":"u1"},"service_tier":"auto",
             "system":[{"type":"text","text":"Be"},{"type":"text","text":"brief.","cache_control":{"type":"ephemeral"}}],
             "messages":[{"role":"user","content":[{"type":"text","text":"Hi"}]},{"role":"assistant","content":"Yo"},{"role":"user","content":"again"}]}
            """);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(
            PromptTokens(RoostChatMessage.System("Be\nbrief."), RoostChatMessage.User("Hi"), RoostChatMessage.Assistant("Yo"), RoostChatMessage.User("again")),
            body.GetProperty("usage").GetProperty("input_tokens").GetInt32());
        Assert.Equal("ok", body.GetProperty("content")[0].GetProperty("text").GetString());
    }

    [Fact]
    public async Task MaxTokensEndsWithStopReasonMaxTokensAndStopSequencesStop()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply), NeverStop = true });

        var (_, body) = await PostAsync(rig.Http, "/v1/messages", """{"model":"roost-tiny-hybrid-test","max_tokens":5,"messages":[{"role":"user","content":"Hi"}]}""");
        Assert.Equal("max_tokens", body.GetProperty("stop_reason").GetString());
        Assert.Equal(5, body.GetProperty("usage").GetProperty("output_tokens").GetInt32());

        var (_, stopped) = await PostAsync(rig.Http, "/v1/messages", Hi + ""","stop_sequences":["world"]}""");
        Assert.Equal("Hello, ", stopped.GetProperty("content")[0].GetProperty("text").GetString());
        Assert.Equal("end_turn", stopped.GetProperty("stop_reason").GetString()); // the client does not say which stop matched
    }

    [Fact]
    public async Task AStreamSendsAnthropicsEventsInOrder()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var response = await rig.Http.PostAsync("/v1/messages", Json(Hi + ""","stream":true}"""), Ct);
        var events = NamedEvents(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        Assert.All(events, e => Assert.Equal(e.Name, e.Data.GetProperty("type").GetString()));
        Assert.Equal(["message_start", "content_block_start", "ping"], events.Take(3).Select(e => e.Name));
        Assert.Equal(["content_block_stop", "message_delta", "message_stop"], events.TakeLast(3).Select(e => e.Name));
        var deltas = events.Skip(3).SkipLast(3).ToList();
        Assert.True(deltas.Count > 3, "text should arrive in several deltas");
        Assert.All(deltas, d =>
        {
            Assert.Equal("content_block_delta", d.Name);
            Assert.Equal(0, d.Data.GetProperty("index").GetInt32());
            Assert.Equal("text_delta", d.Data.GetProperty("delta").GetProperty("type").GetString());
        });
        Assert.Equal(Reply, string.Concat(deltas.Select(d => d.Data.GetProperty("delta").GetProperty("text").GetString())));

        var start = events[0].Data.GetProperty("message");
        Assert.StartsWith("msg_", start.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.Empty(start.GetProperty("content").EnumerateArray());
        Assert.Equal(JsonValueKind.Null, start.GetProperty("stop_reason").ValueKind);
        Assert.Equal("text", events[1].Data.GetProperty("content_block").GetProperty("type").GetString());

        var delta = events[^2].Data;
        Assert.Equal("end_turn", delta.GetProperty("delta").GetProperty("stop_reason").GetString());
        Assert.Equal(ServerRig.Encode(Reply).Length + 1, delta.GetProperty("usage").GetProperty("output_tokens").GetInt32());
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), delta.GetProperty("usage").GetProperty("input_tokens").GetInt32());
    }

    [Fact]
    public async Task APromptLongerThanTheContextIsAnAnthropic400BeforeTheStreamStarts()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/messages",
            $$"""{"model":"roost-tiny-hybrid-test","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"{{new string('x', 400)}}"}]}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertAnthropicError(body, "invalid_request_error", "messages");
    }

    [Theory]
    [InlineData("\"frobnicate\":1", "frobnicate")]
    [InlineData("\"tools\":[{\"name\":\"f\",\"input_schema\":{}}]", "tools")]
    [InlineData("\"tool_choice\":{\"type\":\"auto\"}", "tool_choice")]
    [InlineData("\"thinking\":{\"type\":\"enabled\",\"budget_tokens\":1024}", "thinking")]
    [InlineData("\"temperature\":0.7", "temperature")]
    [InlineData("\"top_p\":0.9", "top_p")]
    [InlineData("\"top_k\":40", "top_k")]
    [InlineData("\"metadata\":{\"other\":\"x\"}", "metadata.other")]
    [InlineData("\"service_tier\":\"priority\"", "service_tier")]
    [InlineData("\"stop_sequences\":\"x\"", "stop_sequences")]
    [InlineData("\"mcp_servers\":[]", "mcp_servers")]
    [InlineData("\"container\":\"c\"", "container")]
    [InlineData("\"max_tokens\":0", "max_tokens")]
    [InlineData("\"stream\":\"yes\"", "stream")]
    public async Task AMessagesOptionIsRefusedByNameInAnthropicsShape(string extra, string param)
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var body = $$"""{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}],{{extra}}}""";
        if (!extra.Contains("max_tokens", StringComparison.Ordinal))
        {
            body = body.Replace("{\"model\"", "{\"max_tokens\":8,\"model\"", StringComparison.Ordinal);
        }

        var (response, error) = await PostAsync(rig.Http, "/v1/messages", body);

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertAnthropicError(error, "invalid_request_error", param);
        Assert.Empty(rig.Executor.Steps); // refused before any work
    }

    [Theory]
    [InlineData("""[{"role":"user","content":[{"type":"image","source":{}}]}]""", "messages[0].content[0].source")]
    [InlineData("""[{"role":"user","content":[{"type":"tool_result"}]}]""", "messages[0].content[0].type")]
    [InlineData("""[{"role":"user","content":[{"type":"text","text":"x","cache_control":{"type":"persistent"}}]}]""", "messages[0].content[0].cache_control")]
    [InlineData("""[{"role":"system","content":"x"}]""", "messages[0].role")]
    [InlineData("""[{"role":"user","content":"x"},{"role":"assistant","content":"prefill"}]""", "messages[1].role")]
    [InlineData("""[{"role":"user"}]""", "messages[0].content")]
    [InlineData("""[{"role":"user","content":"x","name":"n"}]""", "messages[0].name")]
    public async Task AnUnsupportedMessageIsRefusedByPath(string messages, string param)
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, error) = await PostAsync(rig.Http, "/v1/messages", $$"""{"model":"roost-tiny-hybrid-test","max_tokens":8,"messages":{{messages}}}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertAnthropicError(error, "invalid_request_error", param);
    }

    [Fact]
    public async Task MissingFieldsAndBadJsonAreRefusedInAnthropicsShape()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        AssertAnthropicError((await PostAsync(rig.Http, "/v1/messages", """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}]}""")).Body, "invalid_request_error", "max_tokens");
        AssertAnthropicError((await PostAsync(rig.Http, "/v1/messages", """{"max_tokens":8,"messages":[]}""")).Body, "invalid_request_error", "model");
        AssertAnthropicError((await PostAsync(rig.Http, "/v1/messages", """{"model":"other","max_tokens":8,"messages":[{"role":"user","content":"Hi"}]}""")).Body, "invalid_request_error", "model");

        var response = await rig.Http.PostAsync("/v1/messages", Json("{not json"), Ct);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync(Ct));
        AssertAnthropicError(doc.RootElement, "invalid_request_error", null);
    }

    [Fact]
    public async Task CountTokensCountsTheRenderedPrompt()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/messages/count_tokens",
            """{"model":"roost-tiny-hybrid-test","system":"Be brief.","messages":[{"role":"user","content":"Hi"}]}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(PromptTokens(RoostChatMessage.System("Be brief."), RoostChatMessage.User("Hi")), body.GetProperty("input_tokens").GetInt32());
        Assert.Empty(rig.Executor.Steps); // counting runs no step

        var (refused, error) = await PostAsync(rig.Http, "/v1/messages/count_tokens", Hi + "}");
        Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
        AssertAnthropicError(error, "invalid_request_error", "max_tokens");

        var (wrongModel, modelError) = await PostAsync(rig.Http, "/v1/messages/count_tokens", """{"model":"other","messages":[{"role":"user","content":"Hi"}]}""");
        Assert.Equal(HttpStatusCode.BadRequest, wrongModel.StatusCode);
        AssertAnthropicError(modelError, "invalid_request_error", "model");
    }
}
