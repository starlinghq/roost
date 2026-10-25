using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Generation;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;

namespace Starlings.Generation.Roost.Tests;

/// <summary>The chat client end to end over the in-process transport, the C8 rings and the eager worker's plan reader and result writer.</summary>
public sealed class ChatClientTests
{
    // Multi-byte characters so the detokenizer has to hold bytes back between tokens.
    private const string Reply = "Hello, world! 中文 \U0001F600 done.";

    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];

    private static int[] ReplyIds => TestRig.Encode(Reply);

    private static int PromptTokens(params RoostChatMessage[] messages) =>
        TestRig.Encode(Qwen35RoostChatTemplate.Instance.Render(messages)).Length;

    private static CancellationToken Timeout => TestContext.Current.CancellationToken;

    [Fact]
    public async Task GetResponseReturnsTheDecodedTextStopsOnTheStopIdAndReportsUsage()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, executors);

        var response = await client.GetResponseAsync(Hi, cancellationToken: Timeout);

        Assert.Equal(Reply, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(TestRig.ModelId, response.ModelId);
        Assert.NotNull(response.Usage);
        Assert.Equal(PromptTokens(RoostChatMessage.User("Hi")), response.Usage.InputTokenCount);
        Assert.Equal(ReplyIds.Length + 1, response.Usage.OutputTokenCount); // the stop token counts as generated
        Assert.Equal(response.Usage.InputTokenCount + response.Usage.OutputTokenCount, response.Usage.TotalTokenCount);
        Assert.Empty(executors.Single().Rejections);

        // The worker got the model's stop id once per row, and a length budget.
        var decode = executors.Single().Steps.Last();
        Assert.Equal([TestRig.StopId], decode.StopIds);
        Assert.All(executors.Single().Steps, s => Assert.True(s.MaxNewTokensLeft[0] > 0));
    }

    [Fact]
    public async Task StreamingUpdatesConcatenateToTheTextAndTheLastCarriesFinishAndUsage()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds });

        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in client.GetStreamingResponseAsync(Hi, cancellationToken: Timeout))
        {
            updates.Add(update);
        }

        var text = string.Concat(updates.Select(u => u.Text));
        Assert.Equal(Reply, text);
        Assert.True(updates.Count > 3, "text should arrive in several updates");
        Assert.All(updates, u => Assert.DoesNotContain('�', u.Text));
        Assert.All(updates, u => Assert.Equal(ChatRole.Assistant, u.Role));
        Assert.Single(updates.Select(u => u.ResponseId).Distinct());
        Assert.All(updates[..^1], u => Assert.Null(u.FinishReason));
        var last = updates[^1];
        Assert.Equal(ChatFinishReason.Stop, last.FinishReason);
        var usage = Assert.Single(last.Contents.OfType<UsageContent>()).Details;
        Assert.Equal(ReplyIds.Length + 1, usage.OutputTokenCount);
    }

    [Fact]
    public async Task MaxOutputTokensEndsWithLength()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, NeverStop = true }, executors);

        var response = await client.GetResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 5 }, Timeout);

        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        Assert.Equal(5, response.Usage!.OutputTokenCount);
        Assert.Equal(TestRig.Tokenizer.Decode(ReplyIds.AsSpan(0, 5), true), response.Text);
        Assert.Equal(5u, executors.Single().Steps.First(s => s.MaxNewTokensLeft[0] != 0).MaxNewTokensLeft[0]);
    }

    [Fact]
    public async Task APromptLongerThanAStepIsPrefilledInChunks()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, executors);
        var question = string.Concat(Enumerable.Repeat("chunk me ", 12));

        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, question)], cancellationToken: Timeout);

        Assert.Equal(Reply, response.Text);
        var steps = executors.Single().Steps.ToArray();
        int prompt = PromptTokens(RoostChatMessage.User(question));
        Assert.True(prompt > 2 * TestRig.MaxTokensPerStep, $"the prompt ({prompt} tokens) should need three or more chunks");
        Assert.All(steps, s => Assert.True(s.Tokens <= TestRig.MaxTokensPerStep));
        var prefill = steps.TakeWhile(s => s.ChunkTokens[0] > 1).ToArray();
        Assert.Equal(prompt, prefill.Sum(s => s.ChunkTokens[0]));
        Assert.Equal((prompt + TestRig.MaxTokensPerStep - 1) / TestRig.MaxTokensPerStep, prefill.Length);
        Assert.Empty(executors.Single().Rejections);
    }

    [Fact]
    public async Task APromptThatDoesNotFitTheContextIsRefusedAsPromptTooLong()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, executors);
        var question = new string('x', 400);

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, question)], cancellationToken: Timeout));

        Assert.Equal(RoostErrorCode.PromptTooLong, e.Error.Code);
        Assert.Equal("messages", e.Error.Param);
        Assert.Empty(executors.Single().Steps);
    }

    [Fact]
    public async Task OutputIsCappedByTheContextLeft()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, NeverStop = true });
        var question = string.Concat(Enumerable.Repeat("fill ", 35));
        int prompt = PromptTokens(RoostChatMessage.User(question));
        int context = TestRig.AttentionBlocks * 16;
        Assert.InRange(prompt, context - 100, context - 2);

        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, question)], new ChatOptions { MaxOutputTokens = 500 }, Timeout);

        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        Assert.Equal(context - prompt, response.Usage!.OutputTokenCount);
    }

    [Fact]
    public async Task CancellingMidStreamStopsTheRequestAndFreesItsState()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, NeverStop = true, DelayMs = 5 }, executors);
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Timeout);
        var seen = 0;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var update in client.GetStreamingResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 200 }, cts.Token))
            {
                if (++seen == 3)
                {
                    await cts.CancelAsync();
                }
            }
        });

        // The single recurrent slot and the pool are free again: a following request runs to its end.
        var steps = executors.Single().Steps.Count;
        var next = await client.GetResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 4 }, Timeout);
        Assert.Equal(ChatFinishReason.Length, next.FinishReason);
        Assert.True(steps < 100, $"the cancelled request ran {steps} steps; it should have stopped soon after the cancel");
    }

    [Fact]
    public async Task EndingTheEnumerationEarlyCancelsTheRequest()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, NeverStop = true, DelayMs = 2 }, executors);

        await foreach (var update in client.GetStreamingResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 200 }, Timeout))
        {
            break;
        }

        var next = await client.GetResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 3 }, Timeout);
        Assert.Equal(3, next.Usage!.OutputTokenCount);
        Assert.True(executors.Single().Steps.Count < 100);
    }

    [Fact]
    public async Task ConcurrentRequestsAllComplete()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds }, executors);

        var responses = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => client.GetResponseAsync(Hi, cancellationToken: Timeout)));

        Assert.All(responses, r => Assert.Equal(Reply, r.Text));
        Assert.Equal(4, responses.Select(r => r.ResponseId).Distinct().Count());
        Assert.All(executors.Single().Steps, s => Assert.Equal(1, s.Sequences)); // M0: one sequence per step
        Assert.Empty(executors.Single().Rejections);
    }

    [Fact]
    public async Task ARecoverableStepFaultFailsTheRequestAndKeepsTheWorker()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, FaultAtStep = 2, Fault = FaultClass.Recoverable }, executors);

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code);

        var next = await client.GetResponseAsync(Hi, cancellationToken: Timeout);
        Assert.Equal(Reply, next.Text);
        Assert.Single(executors);
    }

    [Fact]
    public async Task AStickyFaultFailsTheRequestAndTheNextRequestStartsANewWorker()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, FaultAtStep = 2, Fault = FaultClass.Sticky }, executors);

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, cancellationToken: Timeout));
        Assert.Equal(RoostErrorCode.DeviceFault, e.Error.Code);

        var next = await client.GetResponseAsync(Hi, cancellationToken: Timeout);
        Assert.Equal(Reply, next.Text);
        Assert.Equal(2, executors.Count);
    }

    [Fact]
    public async Task StopStringsAreRefusedByTheDefaultFactory()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds });

        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(Hi, new ChatOptions { StopSequences = ["world"] }, Timeout));

        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.Equal("stop", e.Error.Param);
        Assert.Equal(RoostCapabilityStatus.Unsupported, client.Capabilities.StatusOf(RoostCapabilities.StopSequences));
    }

    [Fact]
    public async Task StopStringsGoThroughTheSuppliedFactoryAndTruncateTheText()
    {
        var factory = new SimpleStopMatcherFactory();
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds, NeverStop = true }, executors, o => o with { StopMatcherFactory = factory });

        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync(Hi, new ChatOptions { StopSequences = ["中文"], MaxOutputTokens = 100 }, Timeout))
        {
            updates.Add(u);
        }

        Assert.Equal("Hello, world! ", string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(ChatFinishReason.Stop, updates[^1].FinishReason);
        Assert.Equal(["中文"], factory.Created.Single());
        Assert.Equal(RoostCapabilityStatus.Experimental, client.Capabilities.StatusOf(RoostCapabilities.StopSequences));
        Assert.True(updates[^1].Contents.OfType<UsageContent>().Single().Details.OutputTokenCount < 100);

        // The stopped request was dropped from the worker: the next one runs.
        var next = await client.GetResponseAsync(Hi, new ChatOptions { MaxOutputTokens = 2 }, Timeout);
        Assert.Equal(2, next.Usage!.OutputTokenCount);
    }

    [Fact]
    public async Task InstructionsBecomeTheSystemMessage()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds });

        var response = await client.GetResponseAsync(Hi, new ChatOptions { Instructions = "Be brief." }, Timeout);

        Assert.Equal(PromptTokens(RoostChatMessage.System("Be brief."), RoostChatMessage.User("Hi")), response.Usage!.InputTokenCount);
        var e = await Assert.ThrowsAsync<RoostChatException>(() => client.GetResponseAsync(
            [new ChatMessage(ChatRole.System, "x"), new ChatMessage(ChatRole.User, "Hi")], new ChatOptions { Instructions = "Be brief." }, Timeout));
        Assert.Equal("instructions", e.Error.Param);
    }

    [Fact]
    public async Task AWorkerWithOtherWeightsIsRejectedAsAssetMismatch()
    {
        var e = await Assert.ThrowsAsync<RoostChatException>(() => TestRig.InProcessAsync(
            new Script { Tokens = ReplyIds },
            configure: o => o with { Model = o.Model with { WeightsDigest = ScriptedExecutor.ManifestDigest } }));

        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
    }

    [Fact]
    public async Task MetadataServicesAndCapabilities()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds });
        IChatClient chat = client;

        var metadata = chat.GetService<ChatClientMetadata>();
        Assert.NotNull(metadata);
        Assert.Equal("roost", metadata.ProviderName);
        Assert.Equal(TestRig.ModelId, metadata.DefaultModelId);
        Assert.Same(client, chat.GetService<RoostChatClient>());
        Assert.Same(client.Capabilities, chat.GetService<RoostCapabilities>());
        Assert.Null(chat.GetService<ChatClientMetadata>("keyed"));

        var caps = client.Capabilities;
        Assert.Equal(RoostCapabilityStatus.Experimental, caps.StatusOf(RoostCapabilities.Streaming));
        Assert.Equal(RoostCapabilityStatus.Experimental, caps.StatusOf(RoostCapabilities.GreedyDecoding));
        foreach (var name in new[] { RoostCapabilities.Sampling, RoostCapabilities.Tools, RoostCapabilities.ResponseFormat, RoostCapabilities.Logprobs, RoostCapabilities.Penalties, RoostCapabilities.Reasoning, RoostCapabilities.MultimodalInput, RoostCapabilities.ContinuousBatching, RoostCapabilities.WorkerRecovery, "beam_search" })
        {
            Assert.Equal(RoostCapabilityStatus.Unsupported, caps.StatusOf(name));
        }

        Assert.DoesNotContain(caps.Statuses.Values, s => s == RoostCapabilityStatus.Supported);
    }

    [Fact]
    public async Task ADisposedClientRefusesRequests()
    {
        var client = await TestRig.InProcessAsync(new Script { Tokens = ReplyIds });
        await client.DisposeAsync();

        Assert.Throws<ObjectDisposedException>(() => client.GetStreamingResponseAsync(Hi, cancellationToken: TestContext.Current.CancellationToken));
        await client.DisposeAsync();
    }

    [Fact]
    public async Task TheProcessTransportIsTheDefaultAndNeedsACommand()
    {
        Assert.Equal(TransportKind.Process, new RoostWorkerOptions().Transport);
        var e = await Assert.ThrowsAsync<ArgumentException>(() => RoostChatClient.StartAsync(new RoostChatClientOptions { Model = TestRig.Profile() }, Timeout));
        Assert.Contains("Command", e.Message, StringComparison.Ordinal);
    }
}
