using System.Text;
using System.Text.Json;
using OpenAI.Chat;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.ApiCompat.Tests.Support;
using Starlings.Roost.Generation;
using static Starlings.Roost.ApiCompat.Tests.Support.Refusals;

namespace Starlings.Roost.ApiCompat.Tests;

/// <summary><c>POST /v1/chat/completions</c> through the official OpenAI .NET client's <see cref="ChatClient"/>, whole and streamed.</summary>
public sealed class ChatCompletionsCompatTests
{
    // Multi-byte characters so the detokenizer holds bytes back between tokens.
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";

    private static int PromptTokens(params RoostChatMessage[] messages) =>
        ApiCompatRig.Encode(Qwen35RoostChatTemplate.Instance.Render(messages)).Length;

    [Fact]
    public async Task AWholeCompletionParses()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode(Reply) });

        ChatCompletion completion = await rig.Chat.CompleteChatAsync([new UserChatMessage("Hi")], cancellationToken: Ct);

        Assert.StartsWith("chatcmpl-", completion.Id, StringComparison.Ordinal);
        Assert.Equal(ApiCompatRig.ModelId, completion.Model);
        Assert.True(completion.CreatedAt > DateTimeOffset.UnixEpoch);
        Assert.Equal(ChatMessageRole.Assistant, completion.Role);
        Assert.Equal(ChatFinishReason.Stop, completion.FinishReason);
        var part = Assert.Single(completion.Content);
        Assert.Equal(ChatMessageContentPartKind.Text, part.Kind);
        Assert.Equal(Reply, part.Text);
        Assert.Empty(completion.ToolCalls);
        Assert.Null(completion.Refusal);
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), completion.Usage.InputTokenCount);
        Assert.Equal(ApiCompatRig.Encode(Reply).Length + 1, completion.Usage.OutputTokenCount); // the stop token counts
        Assert.Equal(completion.Usage.InputTokenCount + completion.Usage.OutputTokenCount, completion.Usage.TotalTokenCount);
    }

    [Fact]
    public async Task ADeveloperMessageIsTheSystemMessageAndTurnsRenderAsAConversation()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("Fine.") });

        ChatCompletion completion = await rig.Chat.CompleteChatAsync(
            [
                new DeveloperChatMessage("Be brief."),
                new UserChatMessage(ChatMessageContentPart.CreateTextPart("How "), ChatMessageContentPart.CreateTextPart("are you?")),
                new AssistantChatMessage("Well."),
                new UserChatMessage("And now?"),
            ],
            new ChatCompletionOptions { Seed = 7, Temperature = 0, EndUserId = "caller-1" },
            Ct);

        Assert.Equal("Fine.", completion.Content[0].Text);
        var expected = PromptTokens(
            RoostChatMessage.System("Be brief."),
            RoostChatMessage.User("How are you?"),
            RoostChatMessage.Assistant("Well."),
            RoostChatMessage.User("And now?"));
        Assert.Equal(expected, completion.Usage.InputTokenCount);
    }

    [Fact]
    public async Task MaxOutputTokenCountEndsWithFinishReasonLength()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("z"), NeverStop = true });

        ChatCompletion completion = await rig.Chat.CompleteChatAsync(
            [new UserChatMessage("Hi")], new ChatCompletionOptions { MaxOutputTokenCount = 5 }, Ct);

        Assert.Equal(ChatFinishReason.Length, completion.FinishReason);
        Assert.Equal("zzzzz", completion.Content[0].Text);
        Assert.Equal(5, completion.Usage.OutputTokenCount);
    }

    [Fact]
    public async Task AStopSequenceEndsTheTextBeforeIt()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("one, two, three") });

        var options = new ChatCompletionOptions();
        options.StopSequences.Add(", three");
        ChatCompletion completion = await rig.Chat.CompleteChatAsync([new UserChatMessage("Count")], options, Ct);

        Assert.Equal("one, two", completion.Content[0].Text);
        Assert.Equal(ChatFinishReason.Stop, completion.FinishReason);
    }

    [Fact]
    public async Task AStreamedCompletionParsesIntoTextFinishReasonAndUsage()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode(Reply) });

        var updates = new List<StreamingChatCompletionUpdate>();
        await foreach (var update in rig.Chat.CompleteChatStreamingAsync([new UserChatMessage("Hi")], cancellationToken: Ct))
        {
            updates.Add(update);
        }

        Assert.True(updates.Count > 4, "text should arrive in several chunks");
        Assert.Single(updates.Select(u => u.CompletionId).Distinct());
        Assert.StartsWith("chatcmpl-", updates[0].CompletionId, StringComparison.Ordinal);
        Assert.Equal(ChatMessageRole.Assistant, updates[0].Role);
        Assert.Equal(Reply, string.Concat(updates.SelectMany(u => u.ContentUpdate).Select(p => p.Text)));
        Assert.Equal(ChatFinishReason.Stop, Assert.Single(updates, u => u.FinishReason is not null).FinishReason);

        // The client asks for usage on every stream (stream_options.include_usage); it arrives on the last chunk.
        var usage = Assert.Single(updates, u => u.Usage is not null).Usage;
        Assert.Same(usage, updates[^1].Usage);
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), usage.InputTokenCount);
        Assert.Equal(ApiCompatRig.Encode(Reply).Length + 1, usage.OutputTokenCount);
    }

    [Fact]
    public async Task AStreamCutByTheOutputLimitEndsWithFinishReasonLength()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("abcdef") });

        var text = new StringBuilder();
        ChatFinishReason? finish = null;
        await foreach (var update in rig.Chat.CompleteChatStreamingAsync([new UserChatMessage("Hi")], new ChatCompletionOptions { MaxOutputTokenCount = 3 }, Ct))
        {
            text.Append(string.Concat(update.ContentUpdate.Select(p => p.Text)));
            finish ??= update.FinishReason;
        }

        Assert.Equal("abc", text.ToString());
        Assert.Equal(ChatFinishReason.Length, finish);
    }

    [Fact]
    public async Task APromptLongerThanTheContextIsA400PromptTooLongBeforeTheStreamStarts()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("ok") });
        var longPrompt = new string('x', 600); // the context is 256 tokens

        await AssertRefusedAsync(() => rig.Chat.CompleteChatAsync([new UserChatMessage(longPrompt)], cancellationToken: Ct), 400, "prompt_too_long", "messages");
        await AssertRefusedAsync(
            async () =>
            {
                await foreach (var _ in rig.Chat.CompleteChatStreamingAsync([new UserChatMessage(longPrompt)], cancellationToken: Ct))
                {
                }
            },
            400,
            "prompt_too_long",
            "messages");
    }

    [Fact]
    public async Task ReasoningContentIsReadableThroughTheClientsPatch()
    {
        await using var rig = await ApiCompatRig.StartAsync(new Script { Tokens = ApiCompatRig.Encode("Greeting.\n</think>\n\nHello!") });

        // vLLM's chat_template_kwargs is not in the client's model; JsonPatch adds it to the request body.
        var options = new ChatCompletionOptions();
        options.Patch.Set("$.chat_template_kwargs.enable_thinking"u8, true);
        var result = await rig.Chat.CompleteChatAsync([new UserChatMessage("Hi")], options, Ct);
        ChatCompletion completion = result;

        Assert.Equal("Hello!", completion.Content[0].Text);
        Assert.Equal(ChatFinishReason.Stop, completion.FinishReason);
        // reasoning_content (vLLM's field) is outside the client's model: the raw body carries it.
        using var raw = JsonDocument.Parse(result.GetRawResponse().Content.ToString());
        Assert.Equal("Greeting.", raw.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("reasoning_content").GetString());
    }
}
