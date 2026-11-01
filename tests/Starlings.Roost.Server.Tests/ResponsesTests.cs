using System.Net;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary><c>POST /v1/responses</c>, <c>GET /v1/responses/{id}</c> and <c>POST /v1/responses/{id}/cancel</c> over the scripted worker.</summary>
public sealed class ResponsesTests
{
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";
    private const string Model = "\"model\":\"roost-tiny-hybrid-test\"";

    private static int PromptTokens(params RoostChatMessage[] messages) =>
        ServerRig.Encode(Qwen35RoostChatTemplate.Instance.Render(messages)).Length;

    private static string OutputText(JsonElement response)
    {
        var item = Assert.Single(response.GetProperty("output").EnumerateArray());
        Assert.Equal("message", item.GetProperty("type").GetString());
        Assert.Equal("assistant", item.GetProperty("role").GetString());
        var part = Assert.Single(item.GetProperty("content").EnumerateArray());
        Assert.Equal("output_text", part.GetProperty("type").GetString());
        Assert.Empty(part.GetProperty("annotations").EnumerateArray());
        return part.GetProperty("text").GetString()!;
    }

    [Fact]
    public async Task AWholeResponseCompletesAndIsRetrievable()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (response, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","metadata":{"k":"v"},"user":"u1"}""");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        var id = body.GetProperty("id").GetString()!;
        Assert.StartsWith("resp_", id, StringComparison.Ordinal);
        Assert.Equal("response", body.GetProperty("object").GetString());
        Assert.Equal("completed", body.GetProperty("status").GetString());
        Assert.False(body.GetProperty("background").GetBoolean());
        Assert.True(body.GetProperty("store").GetBoolean());
        Assert.Equal(JsonValueKind.Null, body.GetProperty("error").ValueKind);
        Assert.Equal(JsonValueKind.Null, body.GetProperty("incomplete_details").ValueKind);
        Assert.Equal(ServerRig.ModelId, body.GetProperty("model").GetString());
        Assert.Equal("v", body.GetProperty("metadata").GetProperty("k").GetString());
        Assert.Equal("u1", body.GetProperty("user").GetString());
        Assert.Equal("none", body.GetProperty("tool_choice").GetString());
        Assert.Empty(body.GetProperty("tools").EnumerateArray());
        Assert.False(body.GetProperty("parallel_tool_calls").GetBoolean());
        Assert.Equal(Reply, OutputText(body));
        Assert.Equal("completed", body.GetProperty("output")[0].GetProperty("status").GetString());
        var usage = body.GetProperty("usage");
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), usage.GetProperty("input_tokens").GetInt32());
        Assert.Equal(ServerRig.Encode(Reply).Length + 1, usage.GetProperty("output_tokens").GetInt32()); // the stop token counts
        Assert.Equal(usage.GetProperty("input_tokens").GetInt32() + usage.GetProperty("output_tokens").GetInt32(), usage.GetProperty("total_tokens").GetInt32());

        var (got, stored) = await GetAsync(rig.Http, "/v1/responses/" + id);
        Assert.Equal(HttpStatusCode.OK, got.StatusCode);
        Assert.Equal(body.GetRawText(), stored.GetRawText());
    }

    [Fact]
    public async Task InputItemsAndInstructionsRenderAsAConversation()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/responses", $$"""
            {{{Model}},"instructions":"Be brief.","temperature":0,"top_p":1,"tools":[],"tool_choice":"none","parallel_tool_calls":false,
             "text":{"format":{"type":"text"} },"truncation":"disabled","include":[],"service_tier":"auto","prompt_cache_key":"k","stream":false,
             "previous_response_id":null,
             "input":[{"role":"user","content":[{"type":"input_text","text":"H"},{"type":"input_text","text":"i"}]},
                      {"type":"message","id":"msg_1","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Yo","annotations":[]}]},
                      {"role":"user","content":"again"}]}
            """);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("Be brief.", body.GetProperty("instructions").GetString());
        Assert.Equal(
            PromptTokens(RoostChatMessage.System("Be brief."), RoostChatMessage.User("Hi"), RoostChatMessage.Assistant("Yo"), RoostChatMessage.User("again")),
            body.GetProperty("usage").GetProperty("input_tokens").GetInt32());
        Assert.Equal("ok", OutputText(body));
    }

    [Fact]
    public async Task MaxOutputTokensEndsIncomplete()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply), NeverStop = true });

        var (_, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","max_output_tokens":5}""");

        Assert.Equal("incomplete", body.GetProperty("status").GetString());
        Assert.Equal("max_output_tokens", body.GetProperty("incomplete_details").GetProperty("reason").GetString());
        Assert.Equal(5, body.GetProperty("max_output_tokens").GetInt32());
        Assert.Equal("incomplete", body.GetProperty("output")[0].GetProperty("status").GetString());
        Assert.Equal(ServerRig.Tokenizer.Decode(ServerRig.Encode(Reply).AsSpan(0, 5), true), OutputText(body));
        Assert.Equal(5, body.GetProperty("usage").GetProperty("output_tokens").GetInt32());
    }

    [Fact]
    public async Task AResponseNotStoredIsNotFound()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (_, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","store":false}""");
        Assert.False(body.GetProperty("store").GetBoolean());

        var (response, error) = await GetAsync(rig.Http, "/v1/responses/" + body.GetProperty("id").GetString());
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        AssertErrorBody(error, "not_found", "response_id");

        var (cancel, cancelError) = await PostAsync(rig.Http, "/v1/responses/resp_nope/cancel", "{}");
        Assert.Equal(HttpStatusCode.NotFound, cancel.StatusCode);
        AssertErrorBody(cancelError, "not_found", "response_id");
    }

    [Fact]
    public async Task AStreamSendsTheResponsesEventsInOrder()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var response = await rig.Http.PostAsync("/v1/responses", Json($$"""{{{Model}},"input":"Hi","stream":true}"""), Ct);
        var events = NamedEvents(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        Assert.All(events, e => Assert.Equal(e.Name, e.Data.GetProperty("type").GetString()));
        Assert.Equal(Enumerable.Range(0, events.Count), events.Select(e => e.Data.GetProperty("sequence_number").GetInt32()));
        string[] head = ["response.created", "response.in_progress", "response.output_item.added", "response.content_part.added"];
        string[] tail = ["response.output_text.done", "response.content_part.done", "response.output_item.done", "response.completed"];
        Assert.Equal(head, events.Take(4).Select(e => e.Name));
        Assert.Equal(tail, events.TakeLast(4).Select(e => e.Name));
        var deltas = events.Skip(4).SkipLast(4).ToList();
        Assert.True(deltas.Count > 3, "text should arrive in several deltas");
        Assert.All(deltas, d => Assert.Equal("response.output_text.delta", d.Name));
        Assert.Equal(Reply, string.Concat(deltas.Select(d => d.Data.GetProperty("delta").GetString())));

        var created = events[0].Data.GetProperty("response");
        Assert.Equal("in_progress", created.GetProperty("status").GetString());
        Assert.Empty(created.GetProperty("output").EnumerateArray());
        var itemId = events[2].Data.GetProperty("item").GetProperty("id").GetString();
        Assert.All(deltas, d => Assert.Equal(itemId, d.Data.GetProperty("item_id").GetString()));
        Assert.Equal(Reply, events[^4].Data.GetProperty("text").GetString());
        Assert.Equal(Reply, events[^3].Data.GetProperty("part").GetProperty("text").GetString());

        var completed = events[^1].Data.GetProperty("response");
        Assert.Equal("completed", completed.GetProperty("status").GetString());
        Assert.Equal(created.GetProperty("id").GetString(), completed.GetProperty("id").GetString());
        Assert.Equal(Reply, OutputText(completed));

        var (_, stored) = await GetAsync(rig.Http, "/v1/responses/" + completed.GetProperty("id").GetString());
        Assert.Equal(completed.GetRawText(), stored.GetRawText());
    }

    [Fact]
    public async Task AStreamCutByTheOutputLimitEndsWithResponseIncomplete()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply), NeverStop = true });

        var response = await rig.Http.PostAsync("/v1/responses", Json($$"""{{{Model}},"input":"Hi","stream":true,"max_output_tokens":3}"""), Ct);
        var events = NamedEvents(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal("response.incomplete", events[^1].Name);
        Assert.Equal("max_output_tokens", events[^1].Data.GetProperty("response").GetProperty("incomplete_details").GetProperty("reason").GetString());
    }

    [Fact]
    public async Task APromptLongerThanTheContextIsA400BeforeTheStreamStarts()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"stream":true,"input":"{{new string('x', 400)}}"}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "prompt_too_long", "messages");
    }

    [Fact]
    public async Task ABackgroundResponseRunsAndCanBeCancelled()
    {
        // 30 ms a step and no stop token: 200 tokens would take about 6 s.
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("abc"), NeverStop = true, DelayMs = 30 });

        var (response, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","background":true,"max_output_tokens":200}""");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.True(body.GetProperty("background").GetBoolean());
        Assert.True(body.GetProperty("status").GetString() is "queued" or "in_progress");
        var id = body.GetProperty("id").GetString()!;

        // It runs: the stored response gains text while in progress.
        JsonElement running = default;
        for (var i = 0; i < 100; i++)
        {
            (_, running) = await GetAsync(rig.Http, "/v1/responses/" + id);
            if (running.GetProperty("output").GetArrayLength() > 0)
            {
                break;
            }

            await Task.Delay(50, Ct);
        }

        Assert.Equal("in_progress", running.GetProperty("status").GetString());
        Assert.Equal("in_progress", running.GetProperty("output")[0].GetProperty("status").GetString());
        Assert.Equal(JsonValueKind.Null, running.GetProperty("usage").ValueKind);

        var (cancel, cancelled) = await PostAsync(rig.Http, $"/v1/responses/{id}/cancel", string.Empty);
        Assert.Equal(HttpStatusCode.OK, cancel.StatusCode);
        Assert.Equal("cancelled", cancelled.GetProperty("status").GetString());

        int steps = rig.Executor.Steps.Count;
        await Task.Delay(300, Ct);
        Assert.True(rig.Executor.Steps.Count - steps < 3, "the generation ran on after the cancel");

        var (_, after) = await GetAsync(rig.Http, "/v1/responses/" + id);
        Assert.Equal("cancelled", after.GetProperty("status").GetString());

        // Cancelling again answers the cancelled response.
        var (again, _) = await PostAsync(rig.Http, $"/v1/responses/{id}/cancel", string.Empty);
        Assert.Equal(HttpStatusCode.OK, again.StatusCode);
        Assert.Empty(rig.Executor.Rejections);
    }

    [Fact]
    public async Task AFinishedBackgroundResponseCannotBeCancelledAndNorCanAForegroundOne()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (_, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","background":true}""");
        var id = body.GetProperty("id").GetString()!;
        JsonElement done = default;
        for (var i = 0; i < 200; i++)
        {
            (_, done) = await GetAsync(rig.Http, "/v1/responses/" + id);
            if (done.GetProperty("status").GetString() == "completed")
            {
                break;
            }

            await Task.Delay(25, Ct);
        }

        Assert.Equal("completed", done.GetProperty("status").GetString());
        Assert.Equal(Reply, OutputText(done));

        var (cancel, error) = await PostAsync(rig.Http, $"/v1/responses/{id}/cancel", string.Empty);
        Assert.Equal(HttpStatusCode.BadRequest, cancel.StatusCode);
        AssertErrorBody(error, "invalid_request", "response_id");

        var (_, foreground) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi"}""");
        var (cancelForeground, foregroundError) = await PostAsync(rig.Http, $"/v1/responses/{foreground.GetProperty("id").GetString()}/cancel", string.Empty);
        Assert.Equal(HttpStatusCode.BadRequest, cancelForeground.StatusCode);
        AssertErrorBody(foregroundError, "invalid_request", "response_id");
    }

    [Fact]
    public async Task TheStoreIsBoundedAndExpires()
    {
        var clock = new ManualTime();
        await using var rig = await ServerRig.StartAsync(
            new Script { Tokens = ServerRig.Encode("ok") },
            new RoostServerOptions { ResponseStoreCapacity = 2, ResponseStoreTtl = TimeSpan.FromMinutes(10), TimeProvider = clock });

        var ids = new List<string>();
        for (var i = 0; i < 3; i++)
        {
            var (_, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi"}""");
            ids.Add(body.GetProperty("id").GetString()!);
            clock.Advance(TimeSpan.FromMinutes(1));
        }

        // Capacity 2: the oldest made room for the third.
        Assert.Equal(HttpStatusCode.NotFound, (await GetAsync(rig.Http, "/v1/responses/" + ids[0])).Response.StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await GetAsync(rig.Http, "/v1/responses/" + ids[1])).Response.StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await GetAsync(rig.Http, "/v1/responses/" + ids[2])).Response.StatusCode);

        // Ten minutes after creation, a finished response is gone.
        clock.Advance(TimeSpan.FromMinutes(8) + TimeSpan.FromSeconds(1));
        Assert.Equal(HttpStatusCode.NotFound, (await GetAsync(rig.Http, "/v1/responses/" + ids[1])).Response.StatusCode);
        Assert.Equal(HttpStatusCode.OK, (await GetAsync(rig.Http, "/v1/responses/" + ids[2])).Response.StatusCode);
    }

    [Fact]
    public async Task AStoreFullOfRunningResponsesRefusesANewOneWith429()
    {
        await using var rig = await ServerRig.StartAsync(
            new Script { Tokens = ServerRig.Encode("abc"), NeverStop = true, DelayMs = 30 },
            new RoostServerOptions { ResponseStoreCapacity = 1 });

        var (_, first) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","background":true,"max_output_tokens":200}""");
        var (response, error) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi","background":true}""");

        Assert.Equal(HttpStatusCode.TooManyRequests, response.StatusCode);
        AssertErrorBody(error, "queue_full", null);

        var (cancel, _) = await PostAsync(rig.Http, $"/v1/responses/{first.GetProperty("id").GetString()}/cancel", string.Empty);
        Assert.Equal(HttpStatusCode.OK, cancel.StatusCode);
    }

    [Theory]
    [InlineData("\"frobnicate\":1", "frobnicate")]
    [InlineData("\"tools\":[{\"type\":\"function\",\"name\":\"f\"}]", "tools")]
    [InlineData("\"tool_choice\":\"auto\"", "tool_choice")]
    [InlineData("\"reasoning\":{\"effort\":\"low\"}", "reasoning")]
    [InlineData("\"previous_response_id\":\"resp_1\"", "previous_response_id")]
    [InlineData("\"conversation\":\"conv_1\"", "conversation")]
    [InlineData("\"text\":{\"format\":{\"type\":\"json_object\"}}", "text.format")]
    [InlineData("\"text\":{\"verbosity\":\"low\"}", "text.verbosity")]
    [InlineData("\"truncation\":\"auto\"", "truncation")]
    [InlineData("\"include\":[\"reasoning.encrypted_content\"]", "include")]
    [InlineData("\"top_logprobs\":2", "top_logprobs")]
    [InlineData("\"max_tool_calls\":2", "max_tool_calls")]
    [InlineData("\"stream_options\":{\"include_obfuscation\":false}", "stream_options")]
    [InlineData("\"prompt\":{\"id\":\"p\"}", "prompt")]
    [InlineData("\"service_tier\":\"flex\"", "service_tier")]
    [InlineData("\"temperature\":0.7", "temperature")]
    [InlineData("\"top_p\":0.9", "top_p")]
    [InlineData("\"max_output_tokens\":0", "max_output_tokens")]
    [InlineData("\"background\":true,\"store\":false", "background")]
    [InlineData("\"background\":true,\"stream\":true", "background")]
    [InlineData("\"metadata\":{\"k\":1}", "metadata.k")]
    public async Task AResponsesOptionIsRefusedByName(string extra, string param)
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi",{{extra}}}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
        Assert.Empty(rig.Executor.Steps); // refused before any work
    }

    [Theory]
    [InlineData("""[{"type":"function_call_output","call_id":"c","output":"x"}]""", "input[0].type")]
    [InlineData("""[{"role":"user","type":"function_call"}]""", "input[0].type")]
    [InlineData("""[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]""", "input[0].content[0].image_url")]
    [InlineData("""[{"role":"user","content":[{"type":"input_file"}]}]""", "input[0].content[0].type")]
    [InlineData("""[{"role":"assistant","content":[{"type":"input_text","text":"x"}]}]""", "input[0].content[0].type")]
    [InlineData("""[{"role":"tool","content":"x"}]""", "input[0].role")]
    [InlineData("""[{"content":"x"}]""", "input[0].role")]
    [InlineData("3", "input")]
    public async Task AnUnsupportedInputItemIsRefusedByPath(string input, string param)
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var (response, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":{{input}}}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(body, "invalid_option", param);
    }

    [Fact]
    public async Task MissingFieldsAndQueryParametersAreRefusedByName()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        AssertErrorBody((await PostAsync(rig.Http, "/v1/responses", """{"input":"Hi"}""")).Body, "invalid_option", "model");
        AssertErrorBody((await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}}}""")).Body, "invalid_option", "input");
        AssertErrorBody((await PostAsync(rig.Http, "/v1/responses", $$"""{"model":"other","input":"Hi"}""")).Body, "invalid_option", "model");

        var (_, body) = await PostAsync(rig.Http, "/v1/responses", $$"""{{{Model}},"input":"Hi"}""");
        var (response, error) = await GetAsync(rig.Http, $"/v1/responses/{body.GetProperty("id").GetString()}?stream=true");
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        AssertErrorBody(error, "invalid_option", "stream");
    }
}
