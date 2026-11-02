using System.Text;
using OpenAI.Responses;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;
using Starlings.Roost.Generation;
using static Starlings.Roost.ApiCompat.Tests.Support.Refusals;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary>
/// <c>POST /v1/responses</c>, <c>GET /v1/responses/{id}</c> and <c>POST /v1/responses/{id}/cancel</c> (R-074) through the
/// official client's <see cref="ResponsesClient"/>: whole, streamed, background and cancelled.
/// </summary>
public sealed class ResponsesCompatTests
{
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";

    private static int PromptTokens(params RoostChatMessage[] messages) =>
        ApiCompatRig.Encode(Qwen35RoostChatTemplate.Instance.Render(messages)).Length;

    [Fact]
    public async Task AWholeResponseParsesAndIsRetrievable()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode(Reply) });

        var options = new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]) { EndUserId = "u1" };
        options.Metadata["k"] = "v";
        ResponseResult response = await rig.Responses.CreateResponseAsync(options, Ct);

        Assert.StartsWith("resp_", response.Id, StringComparison.Ordinal);
        Assert.Equal(ResponseStatus.Completed, response.Status);
        Assert.Equal(ApiCompatRig.ModelId, response.Model);
        Assert.Null(response.Error);
        Assert.Null(response.IncompleteStatusDetails);
        Assert.Equal("v", response.Metadata["k"]);
        Assert.Equal(Reply, response.GetOutputText());
        var message = Assert.IsType<MessageResponseItem>(Assert.Single(response.OutputItems), exactMatch: false);
        Assert.Equal(MessageRole.Assistant, message.Role);
        Assert.Equal(MessageStatus.Completed, message.Status);
        Assert.Equal(ResponseContentPartKind.OutputText, Assert.Single(message.Content).Kind);
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), response.Usage.InputTokenCount);
        Assert.Equal(ApiCompatRig.Encode(Reply).Length + 1, response.Usage.OutputTokenCount); // the stop token counts
        Assert.Equal(response.Usage.InputTokenCount + response.Usage.OutputTokenCount, response.Usage.TotalTokenCount);

        ResponseResult stored = await rig.Responses.GetResponseAsync(response.Id, Ct);
        Assert.Equal(response.Id, stored.Id);
        Assert.Equal(ResponseStatus.Completed, stored.Status);
        Assert.Equal(Reply, stored.GetOutputText());
    }

    [Fact]
    public async Task InstructionsAndInputItemsRenderAsAConversation()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("ok") });

        var options = new CreateResponseOptions(
            ApiCompatRig.ModelId,
            [
                ResponseItem.CreateUserMessageItem([ResponseContentPart.CreateInputTextPart("H"), ResponseContentPart.CreateInputTextPart("i")]),
                ResponseItem.CreateAssistantMessageItem("Yo"),
                ResponseItem.CreateUserMessageItem("again"),
            ])
        {
            Instructions = "Be brief.",
            Temperature = 0,
            TopP = 1,
        };
        ResponseResult response = await rig.Responses.CreateResponseAsync(options, Ct);

        Assert.Equal("ok", response.GetOutputText());
        Assert.Equal(
            PromptTokens(RoostChatMessage.System("Be brief."), RoostChatMessage.User("Hi"), RoostChatMessage.Assistant("Yo"), RoostChatMessage.User("again")),
            response.Usage.InputTokenCount);
    }

    [Fact]
    public async Task MaxOutputTokenCountEndsIncomplete()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("z"), NeverStop = true });

        ResponseResult response = await rig.Responses.CreateResponseAsync(
            new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]) { MaxOutputTokenCount = 5 }, Ct);

        Assert.Equal(ResponseStatus.Incomplete, response.Status);
        Assert.Equal(ResponseIncompleteStatusReason.MaxOutputTokens, response.IncompleteStatusDetails.Reason);
        Assert.Equal("zzzzz", response.GetOutputText());
        Assert.Equal(5, response.Usage.OutputTokenCount);
    }

    [Fact]
    public async Task AStreamParsesIntoTheResponsesEvents()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode(Reply) });

        var updates = new List<StreamingResponseUpdate>();
        await foreach (var update in rig.Responses.CreateResponseStreamingAsync(
            new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]) { StreamingEnabled = true }, Ct))
        {
            updates.Add(update);
        }

        Assert.IsType<StreamingResponseCreatedUpdate>(updates[0]);
        Assert.IsType<StreamingResponseInProgressUpdate>(updates[1]);
        Assert.IsType<StreamingResponseOutputItemAddedUpdate>(updates[2]);
        Assert.IsType<StreamingResponseContentPartAddedUpdate>(updates[3]);
        var completed = Assert.IsType<StreamingResponseCompletedUpdate>(updates[^1]);
        Assert.IsType<StreamingResponseOutputItemDoneUpdate>(updates[^2]);
        Assert.IsType<StreamingResponseContentPartDoneUpdate>(updates[^3]);
        var done = Assert.IsType<StreamingResponseOutputTextDoneUpdate>(updates[^4]);
        Assert.Equal(Enumerable.Range(0, updates.Count), updates.Select(u => u.SequenceNumber));

        var deltas = updates.OfType<StreamingResponseOutputTextDeltaUpdate>().ToList();
        Assert.True(deltas.Count > 3, "text should arrive in several deltas");
        Assert.Equal(Reply, string.Concat(deltas.Select(d => d.Delta)));
        Assert.Equal(Reply, done.Text);
        Assert.Equal(ResponseStatus.Completed, completed.Response.Status);
        Assert.Equal(Reply, completed.Response.GetOutputText());
        Assert.Equal(ApiCompatRig.Encode(Reply).Length + 1, completed.Response.Usage.OutputTokenCount);

        ResponseResult stored = await rig.Responses.GetResponseAsync(completed.Response.Id, Ct);
        Assert.Equal(Reply, stored.GetOutputText());
    }

    [Fact]
    public async Task AStreamCutByTheOutputLimitEndsWithResponseIncomplete()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("abcdef") });

        var text = new StringBuilder();
        StreamingResponseUpdate? last = null;
        await foreach (var update in rig.Responses.CreateResponseStreamingAsync(
            new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]) { MaxOutputTokenCount = 3, StreamingEnabled = true }, Ct))
        {
            text.Append((update as StreamingResponseOutputTextDeltaUpdate)?.Delta);
            last = update;
        }

        Assert.Equal("abc", text.ToString());
        var incomplete = Assert.IsType<StreamingResponseIncompleteUpdate>(last);
        Assert.Equal(ResponseStatus.Incomplete, incomplete.Response.Status);
        Assert.Equal(ResponseIncompleteStatusReason.MaxOutputTokens, incomplete.Response.IncompleteStatusDetails.Reason);
    }

    [Fact]
    public async Task ABackgroundResponseRunsAndIsCancelled()
    {
        // 30 ms a step and no stop token: 200 tokens would take about 6 s.
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("abc"), NeverStop = true, DelayMs = 30 });

        ResponseResult created = await rig.Responses.CreateResponseAsync(
            new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]) { BackgroundModeEnabled = true, MaxOutputTokenCount = 200 }, Ct);
        Assert.True(created.BackgroundModeEnabled);
        Assert.True(created.Status is ResponseStatus.Queued or ResponseStatus.InProgress);

        ResponseResult running = created;
        for (var i = 0; i < 100 && running.OutputItems.Count == 0; i++)
        {
            await Task.Delay(50, Ct);
            running = await rig.Responses.GetResponseAsync(created.Id, Ct);
        }

        Assert.Equal(ResponseStatus.InProgress, running.Status);

        ResponseResult cancelled = await rig.Responses.CancelResponseAsync(created.Id, Ct);
        Assert.Equal(ResponseStatus.Cancelled, cancelled.Status);
        ResponseResult after = await rig.Responses.GetResponseAsync(created.Id, Ct);
        Assert.Equal(ResponseStatus.Cancelled, after.Status);
    }

    [Fact]
    public async Task MissingAndForegroundResponsesAreRefusedWithTheErrorBody()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("ok") });

        await AssertRefusedAsync(() => rig.Responses.GetResponseAsync("resp_nope", Ct), 404, "not_found", "response_id");
        await AssertRefusedAsync(() => rig.Responses.CancelResponseAsync("resp_nope", Ct), 404, "not_found", "response_id");

        ResponseResult unstored = await rig.Responses.CreateResponseAsync(
            new CreateResponseOptions(ApiCompatRig.ModelId, [ResponseItem.CreateUserMessageItem("Hi")]) { StoredOutputEnabled = false }, Ct);
        await AssertRefusedAsync(() => rig.Responses.GetResponseAsync(unstored.Id, Ct), 404, "not_found", "response_id");

        ResponseResult foreground = await rig.Responses.CreateResponseAsync(ApiCompatRig.ModelId, "Hi", cancellationToken: Ct);
        await AssertRefusedAsync(() => rig.Responses.CancelResponseAsync(foreground.Id, Ct), 400, "invalid_request", "response_id");
    }
}
