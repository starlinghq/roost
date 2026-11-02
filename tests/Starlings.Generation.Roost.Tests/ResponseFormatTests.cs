using System.Text.Json;
using Microsoft.Extensions.AI;
using Starlings.Generation.Roost.Tests.Support;
using Starlings.Roost.Native;
using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost.Tests;

/// <summary>
/// R-051 response formats through <see cref="RoostChatClient"/>: a JSON or JSON-schema format is compiled with the grammar engine
/// (an invalid schema is the caller's 400 naming <c>response_format</c>), and is then refused, because the worker does not take
/// token masks yet. Tests that compile skip when the roost-grammar library is not built.
/// </summary>
public sealed class ResponseFormatTests : IAsyncLifetime
{
    private static readonly ChatMessage[] Hi = [new(ChatRole.User, "Hi")];
    private static readonly Lazy<(GrammarTokenizer? Tokenizer, string Reason)> s_grammar = new(LoadGrammar);
    private readonly List<ScriptedExecutor> _executors = [];
    private RoostChatClient? _client;

    public async ValueTask InitializeAsync() =>
        _client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") }, _executors, o => o with { Grammar = s_grammar.Value.Tokenizer });

    public async ValueTask DisposeAsync()
    {
        if (_client is not null)
        {
            await _client.DisposeAsync();
        }
    }

    [Fact]
    public void ACompilableFormatIsRefusedBecauseTheWorkerTakesNoMasksYet()
    {
        Require();
        foreach (var format in new[] { ChatResponseFormat.Json, Schema("""{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}""") })
        {
            var e = Refused(new ChatOptions { ResponseFormat = format });
            Assert.Contains("cannot be applied yet", e.Error.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void AnUnsupportedSchemaKeywordIsTheCallersFault()
    {
        Require();
        var e = Refused(new ChatOptions { ResponseFormat = Schema("""{"type":"array","uniqueItems":true}""") });
        Assert.Contains("cannot be enforced", e.Error.Message, StringComparison.Ordinal);
        Assert.Contains("uniqueItems", e.Error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASchemaThatIsNotAnObjectIsRefused()
    {
        var e = Refused(new ChatOptions { ResponseFormat = Schema("[1]") });
        Assert.Contains("must be an object", e.Error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheRawCompletionChecksTheFormatToo()
    {
        Require();
        var options = new ChatOptions { ResponseFormat = Schema("""{"type":"string","uniqueItems":true}""") };
        var e = Assert.Throws<RoostChatException>(() => _client!.GetStreamingCompletionAsync("x", options, TestContext.Current.CancellationToken));
        Assert.Equal("response_format", e.Error.Param);
        Assert.Contains("uniqueItems", e.Error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task WithoutTheGrammarEngineEveryJsonFormatIsRefused()
    {
        await using var client = await TestRig.InProcessAsync(new Script { Tokens = TestRig.Encode("ok") });
        var e = Assert.Throws<RoostChatException>(() => client.GetStreamingResponseAsync(Hi, new ChatOptions { ResponseFormat = ChatResponseFormat.Json }, TestContext.Current.CancellationToken));
        Assert.Equal("response_format", e.Error.Param);
        Assert.Contains("grammar engine", e.Error.Message, StringComparison.Ordinal);
        Assert.Equal(RoostCapabilityStatus.Unsupported, client.Capabilities.StatusOf(RoostCapabilities.ResponseFormat));
    }

    private static ChatResponseFormatJson Schema(string json)
    {
        using var document = JsonDocument.Parse(json);
        return ChatResponseFormat.ForJsonSchema(document.RootElement.Clone(), "answer");
    }

    private static void Require() => Assert.SkipUnless(s_grammar.Value.Tokenizer is not null, s_grammar.Value.Reason);

    private static (GrammarTokenizer?, string) LoadGrammar()
    {
        var configured = Environment.GetEnvironmentVariable(RoostGrammarLibrary.PathVariable);
        string? path = !string.IsNullOrEmpty(configured) ? configured : null;
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); path is null && dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Roost.slnx")))
            {
                var built = Path.Combine(dir.FullName, "native", "roost-grammar", "target", "release", RoostGrammarLibrary.FileName);
                path = File.Exists(built) ? built : null;
                break;
            }
        }

        if (path is null)
        {
            return (null, $"roost-grammar is not built: run `cargo build --release --locked` in native/roost-grammar, or set {RoostGrammarLibrary.PathVariable}");
        }

        try
        {
            var json = File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json"));
            return (RoostGrammarLibrary.Load(path).CreateTokenizer(json, 0, [TestRig.StopId]), "");
        }
        catch (GrammarException e)
        {
            return (null, e.Message);
        }
    }

    private RoostChatException Refused(ChatOptions options)
    {
        var e = Assert.Throws<RoostChatException>(() => _client!.GetStreamingResponseAsync(Hi, options, TestContext.Current.CancellationToken));
        Assert.Equal(RoostErrorCode.InvalidOption, e.Error.Code);
        Assert.Equal("response_format", e.Error.Param);
        Assert.Empty(_executors.Single().Steps); // refused before any work
        return e;
    }
}
