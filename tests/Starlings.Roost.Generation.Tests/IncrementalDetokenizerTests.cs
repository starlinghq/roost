using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Roost.Generation.Tests;

public sealed class IncrementalDetokenizerTests
{
    // Profile qwen35-bpe-chat of Starlings.Tokenizers 0.3.0 (asset sha256 and size from profiles/qwen35-bpe-chat.yaml).
    private const string QwenTokenizerSha256 = "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f";
    private const string QwenTokenizerEnv = "ROOST_QWEN35_TOKENIZER_JSON";

    public static readonly string[] Texts =
    [
        "",
        "plain ascii text, the quick brown fox",
        " the then there",
        "中文，日本語のテキスト",
        "emoji \U0001F600\U0001F680 and flags \U0001F1EB\U0001F1F7",
        "combining é ạ̈ and ü",
        "mixed 中 the \U0001F600 end\n\n\ttabs  spaces  ",
        "<|im_end|> after a special token",
    ];

    public static TheoryData<string> TextData() => [.. Texts];

    private static Tokenizer LoadTiny()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json");
        return Tokenizer.FromFile(path);
    }

    private static string Stream(RoostIncrementalDetokenizer detokenizer, int[] ids, List<string>? pieces = null)
    {
        var text = new System.Text.StringBuilder();
        foreach (var id in ids)
        {
            var piece = detokenizer.Push(id);
            pieces?.Add(piece);
            text.Append(piece);
        }

        text.Append(detokenizer.Flush());
        return text.ToString();
    }

    [Theory]
    [MemberData(nameof(TextData))]
    public void RoundTripsOnSmallByteLevelTokenizer(string text)
    {
        var tokenizer = LoadTiny();
        var ids = tokenizer.Encode(text, false).Ids.ToArray();
        var pieces = new List<string>();
        var streamed = Stream(new RoostIncrementalDetokenizer(tokenizer, skipSpecialTokens: false), ids, pieces);

        Assert.Equal(tokenizer.Decode(ids, false), streamed);
        Assert.Equal(text.Normalize(System.Text.NormalizationForm.FormC), streamed); // the tokenizer normalises to NFC
        Assert.All(pieces, p => Assert.DoesNotContain('�', p));
    }

    [Fact]
    public void NeverEmitsHalfACharacter()
    {
        var tokenizer = LoadTiny();
        var ids = tokenizer.Encode("\U0001F600", false).Ids.ToArray();
        Assert.True(ids.Length >= 2, "an emoji must span several byte-level tokens in this fixture");

        var detokenizer = new RoostIncrementalDetokenizer(tokenizer, false);
        for (var i = 0; i < ids.Length - 1; i++)
        {
            Assert.Equal("", detokenizer.Push(ids[i]));
        }

        Assert.Equal("\U0001F600", detokenizer.Push(ids[^1]));
        Assert.Equal("", detokenizer.Flush());
    }

    [Fact]
    public void FlushEmitsReplacementForATruncatedSequenceAndResets()
    {
        var tokenizer = LoadTiny();
        var ids = tokenizer.Encode("\U0001F600", false).Ids.ToArray();
        var detokenizer = new RoostIncrementalDetokenizer(tokenizer, false);
        detokenizer.Push(ids[0]);
        Assert.Equal(1, detokenizer.Count);
        Assert.Contains('�', detokenizer.Flush());
        Assert.Equal(0, detokenizer.Count);
        Assert.Equal("ok", Stream(detokenizer, tokenizer.Encode("ok", false).Ids.ToArray()));
    }

    [Fact]
    public void SkipsSpecialTokensWhenAsked()
    {
        var tokenizer = LoadTiny();
        var ids = tokenizer.Encode("a<|im_end|>b", false).Ids.ToArray();
        Assert.Equal("ab", Stream(new RoostIncrementalDetokenizer(tokenizer, skipSpecialTokens: true), ids));
        Assert.Equal("a<|im_end|>b", Stream(new RoostIncrementalDetokenizer(tokenizer, skipSpecialTokens: false), ids));
    }

    [Fact]
    public void MatchesStarlingDetokenizerPieceForPiece()
    {
        var tokenizer = LoadTiny();
        var ids = tokenizer.Encode(Texts[6], false).Ids.ToArray();
        var ours = new RoostIncrementalDetokenizer(tokenizer, true);
        var theirs = new Starlings.Generation.OnnxGenAI.IncrementalDetokenizer(tokenizer, true);
        foreach (var id in ids)
        {
            Assert.Equal(theirs.Push(id), ours.Push(id));
        }

        Assert.Equal(theirs.Flush(), ours.Flush());
    }

    [Fact]
    public void NullTokenizerIsRejected() =>
        Assert.Throws<ArgumentNullException>(() => new RoostIncrementalDetokenizer(null!));

    [Theory]
    [MemberData(nameof(TextData))]
    public void RoundTripsOnQwen35Tokenizer(string text)
    {
        var path = Environment.GetEnvironmentVariable(QwenTokenizerEnv);
        if (string.IsNullOrEmpty(path))
        {
            Assert.Skip($"Set {QwenTokenizerEnv} to the qwen35-bpe-chat tokenizer.json (sha256 {QwenTokenizerSha256}) to run this case.");
        }

        var digest = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        Assert.Equal(QwenTokenizerSha256, digest);

        var tokenizer = Tokenizer.FromFile(path);
        var ids = tokenizer.Encode(text, false).Ids.ToArray();
        var streamed = Stream(new RoostIncrementalDetokenizer(tokenizer, skipSpecialTokens: false), ids);
        Assert.Equal(tokenizer.Decode(ids, false), streamed);
        Assert.Equal(text.Normalize(System.Text.NormalizationForm.FormC), streamed); // the tokenizer normalises to NFC
    }
}
