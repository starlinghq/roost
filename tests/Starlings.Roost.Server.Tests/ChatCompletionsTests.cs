using System.Net;
using System.Text;
using System.Text.Json;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Server.Tests.Support;
using static Starlings.Roost.Server.Tests.Support.Http;

namespace Starlings.Roost.Server.Tests;

/// <summary><c>POST /v1/chat/completions</c> end to end: HTTP, the chat client, the C8 rings and the eager worker's plan reader.</summary>
public sealed class ChatCompletionsTests
{
    // Multi-byte characters so the detokenizer holds bytes back between tokens.
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";

    private const string Hi = """{"model":"roost-tiny-hybrid-test","messages":[{"role":"user","content":"Hi"}]""";

    private static int PromptTokens(string user) =>
        ServerRig.Encode(Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.User(user)])).Length;

    [Fact]
    public async Task ANonStreamingRequestReturnsAChatCompletion()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + "}");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        Assert.Equal("chat.completion", body.GetProperty("object").GetString());
        Assert.StartsWith("chatcmpl-", body.GetProperty("id").GetString(), StringComparison.Ordinal);
        Assert.Equal(ServerRig.ModelId, body.GetProperty("model").GetString());
        Assert.True(body.GetProperty("created").GetInt64() > 0);
        var choice = Assert.Single(body.GetProperty("choices").EnumerateArray());
        Assert.Equal(0, choice.GetProperty("index").GetInt32());
        Assert.Equal("assistant", choice.GetProperty("message").GetProperty("role").GetString());
        Assert.Equal(Reply, choice.GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
        Assert.Equal(JsonValueKind.Null, choice.GetProperty("logprobs").ValueKind);
        var usage = body.GetProperty("usage");
        Assert.Equal(PromptTokens("Hi"), usage.GetProperty("prompt_tokens").GetInt32());
        Assert.Equal(ServerRig.Encode(Reply).Length + 1, usage.GetProperty("completion_tokens").GetInt32()); // the stop token counts
        Assert.Equal(usage.GetProperty("prompt_tokens").GetInt32() + usage.GetProperty("completion_tokens").GetInt32(), usage.GetProperty("total_tokens").GetInt32());
        Assert.Empty(rig.Executor.Rejections);
    }

    [Fact]
    public async Task AStreamingRequestSendsChunksThenUsageThenDone()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var response = await rig.Http.PostAsync("/v1/chat/completions", Json(Hi + ""","stream":true,"stream_options":{"include_usage":true}}"""), Ct);
        var events = Events(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        Assert.Equal("[DONE]", events[^1]);
        var chunks = events[..^1].Select(e => JsonDocument.Parse(e).RootElement).ToList();
        Assert.All(chunks, c => Assert.Equal("chat.completion.chunk", c.GetProperty("object").GetString()));
        Assert.Single(chunks.Select(c => c.GetProperty("id").GetString()).Distinct());

        var first = chunks[0].GetProperty("choices")[0];
        Assert.Equal("assistant", first.GetProperty("delta").GetProperty("role").GetString());

        var usageChunk = chunks[^1];
        Assert.Empty(usageChunk.GetProperty("choices").EnumerateArray());
        Assert.Equal(ServerRig.Encode(Reply).Length + 1, usageChunk.GetProperty("usage").GetProperty("completion_tokens").GetInt32());

        var finishChunk = chunks[^2].GetProperty("choices")[0];
        Assert.Equal("stop", finishChunk.GetProperty("finish_reason").GetString());
        Assert.False(finishChunk.GetProperty("delta").TryGetProperty("content", out _));

        var content = chunks[1..^2].Select(c => c.GetProperty("choices")[0]).ToList();
        Assert.True(content.Count > 3, "text should arrive in several chunks");
        Assert.All(content, c => Assert.Equal(JsonValueKind.Null, c.GetProperty("finish_reason").ValueKind));
        Assert.All(chunks[..^1], c => Assert.False(c.TryGetProperty("usage", out _)));
        Assert.Equal(Reply, string.Concat(content.Select(c => c.GetProperty("delta").GetProperty("content").GetString())));
    }

    [Fact]
    public async Task AStreamWithoutIncludeUsageEndsWithTheFinishChunk()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("ok") });

        var response = await rig.Http.PostAsync("/v1/chat/completions", Json(Hi + ""","stream":true}"""), Ct);
        var events = Events(await response.Content.ReadAsStringAsync(Ct));

        Assert.Equal("[DONE]", events[^1]);
        var last = JsonDocument.Parse(events[^2]).RootElement;
        Assert.Equal("stop", last.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.DoesNotContain(events[..^1], e => JsonDocument.Parse(e).RootElement.TryGetProperty("usage", out _));
    }

    [Fact]
    public async Task MaxTokensEndsWithLengthAndTextPartsAndNullsAreAccepted()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply), NeverStop = true });

        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions", """
            {"model":"roost-tiny-hybrid-test","max_completion_tokens":5,"temperature":0,"top_p":1,"n":1,"seed":7,"user":"u1",
             "tools":null,"logprobs":false,"stream":false,"response_format":{"type":"text"},"tool_choice":"none",
             "messages":[{"role":"developer","content":"Be brief."},{"role":"user","content":[{"type":"text","text":"H"},{"type":"text","text":"i"}]}]}
            """);

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var choice = body.GetProperty("choices")[0];
        Assert.Equal("length", choice.GetProperty("finish_reason").GetString());
        Assert.Equal(ServerRig.Tokenizer.Decode(ServerRig.Encode(Reply).AsSpan(0, 5), true), choice.GetProperty("message").GetProperty("content").GetString());
        Assert.Equal(5, body.GetProperty("usage").GetProperty("completion_tokens").GetInt32());
        var expectedPrompt = ServerRig.Encode(Qwen35RoostChatTemplate.Instance.Render([RoostChatMessage.System("Be brief."), RoostChatMessage.User("Hi")])).Length;
        Assert.Equal(expectedPrompt, body.GetProperty("usage").GetProperty("prompt_tokens").GetInt32());
    }

    [Fact]
    public async Task StopStringsEndTheReplyWithStop()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });

        var (_, body) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + ""","stop":["world"]}""");

        var choice = body.GetProperty("choices")[0];
        Assert.Equal("Hello, ", choice.GetProperty("message").GetProperty("content").GetString());
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
    }

    [Fact]
    public async Task APromptLongerThanTheContextIsA400BeforeTheStreamStarts()
    {
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode(Reply) });
        var question = new string('x', 400);

        var (response, body) = await PostAsync(rig.Http, "/v1/chat/completions",
            $$"""{"model":"roost-tiny-hybrid-test","stream":true,"messages":[{"role":"user","content":"{{question}}"}]}""");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("application/json", response.Content.Headers.ContentType?.MediaType);
        var error = AssertErrorBody(body, "prompt_too_long", "messages");
        Assert.DoesNotContain(question, error.GetProperty("message").GetString(), StringComparison.Ordinal); // no prompt text in errors
    }

    [Fact]
    public async Task ADisconnectedClientCancelsTheGeneration()
    {
        // 30 ms a step and no stop token: 200 tokens would take about 6 s if the request ran on.
        await using var rig = await ServerRig.StartAsync(new Script { Tokens = ServerRig.Encode("abc"), NeverStop = true, DelayMs = 30 });
        using (var request = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions"))
        {
            request.Content = Json(Hi + ""","stream":true,"max_tokens":200}""");
            using var response = await rig.Http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, Ct);
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            var stream = await response.Content.ReadAsStreamAsync(Ct);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            var contentEvents = 0;
            while (contentEvents < 2 && await reader.ReadLineAsync(Ct) is { } line)
            {
                contentEvents += line.Contains("\"content\":\"", StringComparison.Ordinal) && !line.Contains("\"role\"", StringComparison.Ordinal) ? 1 : 0;
            }

            Assert.Equal(2, contentEvents);
        } // disposing the response aborts the request: the server's RequestAborted fires

        // The worker stops getting steps for the sequence.
        int settled = await SettledStepCountAsync(rig);
        Assert.True(settled < 100, $"the generation ran on after the disconnect ({settled} steps)");

        // And the driver is free for the next request at once (M0 runs one sequence per step).
        var (next, body) = await PostAsync(rig.Http, "/v1/chat/completions", Hi + ""","max_tokens":2}""");
        Assert.Equal(HttpStatusCode.OK, next.StatusCode);
        Assert.Equal("length", body.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Empty(rig.Executor.Rejections);
    }

    private static async Task<int> SettledStepCountAsync(ServerRig rig)
    {
        var last = -1;
        for (var i = 0; i < 50; i++)
        {
            await Task.Delay(200, Ct);
            var now = rig.Executor.Steps.Count;
            if (now == last)
            {
                return now;
            }

            last = now;
        }

        Assert.Fail("the step count never settled");
        return last;
    }
}
