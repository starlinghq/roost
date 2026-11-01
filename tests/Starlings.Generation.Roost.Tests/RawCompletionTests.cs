using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>Raw-text completion (the server's completions route, R-037): the prompt is tokenised as given, with no chat template.</summary>
public sealed class RawCompletionTests
{
    private const string Reply = "four";

    private static CancellationToken Timeout => TestContext.Current.CancellationToken;

    [Fact]
    public async Task ThePromptIsTokenisedAsGivenWithoutTheChatTemplate()
    {
        var executors = new List<ScriptedExecutor>();
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode(Reply) }, executors);
        const string prompt = "two plus two is ";

        var response = await client.GetCompletionAsync(prompt, cancellationToken: Timeout);

        Assert.Equal(Reply, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(TestRig.Encode(prompt).Length, response.Usage!.InputTokenCount);
        var prefill = executors.Single().Steps.First();
        Assert.Equal(TestRig.Encode(prompt).Length, prefill.ChunkTokens[0]);
        Assert.Empty(executors.Single().Rejections);
    }

    [Fact]
    public async Task StreamingAndOptionsBehaveAsForAChat()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("abcdef"), NeverStop = true });

        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingCompletionAsync("x", new ChatOptions { MaxOutputTokens = 3 }, Timeout))
        {
            updates.Add(u);
        }

        Assert.Equal("abc", string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(ChatFinishReason.Length, updates[^1].FinishReason);
    }

    [Theory]
    [InlineData("instructions")]
    [InlineData("temperature")]
    [InlineData("prompt")]
    public async Task RefusalsNameTheOption(string param)
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode(Reply) });
        var (prompt, options) = param switch
        {
            "instructions" => ("x", new ChatOptions { Instructions = "be brief" }),
            "temperature" => ("x", new ChatOptions { Temperature = 0.7f }),
            _ => (string.Empty, (ChatOptions?)null),
        };

        var e = Assert.Throws<RoostChatException>(() => client.GetStreamingCompletionAsync(prompt, options, Timeout));

        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.Equal(param, e.Error.Param);
    }

    [Fact]
    public async Task TheCapabilityIsUntested()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode(Reply) });

        Assert.Equal(RoostCapabilityStatus.Untested, client.Capabilities.StatusOf(RoostCapabilities.RawCompletion));
    }
}
