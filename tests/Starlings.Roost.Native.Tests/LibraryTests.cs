using Starlings.Roost.Native.Tests.Support;

namespace Starlings.Roost.Native.Tests;

/// <summary>Loading roost-grammar: the ABI rule, the lookup order and the errors for a file that is not the library.</summary>
public sealed class LibraryTests
{
    [Fact]
    public void FileNameFollowsThePlatform()
    {
        var name = RoostGrammarLibrary.FileName;
        if (OperatingSystem.IsWindows())
        {
            Assert.Equal("roost_grammar.dll", name);
        }
        else if (OperatingSystem.IsMacOS())
        {
            Assert.Equal("libroost_grammar.dylib", name);
        }
        else
        {
            Assert.Equal("libroost_grammar.so", name);
        }
    }

    [Fact]
    public void AMissingFileIsAGrammarException()
    {
        var missing = Path.Combine(AppContext.BaseDirectory, "no-such-dir", RoostGrammarLibrary.FileName);
        var e = Assert.Throws<GrammarException>(() => RoostGrammarLibrary.Load(missing));
        Assert.Contains("could not be loaded", e.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileThatIsNotALibraryIsAGrammarException()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json");
        _ = Assert.Throws<GrammarException>(() => RoostGrammarLibrary.Load(path));
    }

    [Fact]
    public void TheBuiltLibraryHasAbiOneZero()
    {
        _ = Grammar.Require();
        var library = Grammar.Library!;
        Assert.Equal(1u << 16, library.AbiVersion);
        Assert.Equal(RoostGrammarLibrary.AbiMajor, (int)(library.AbiVersion >> 16));
        Assert.Same(library, RoostGrammarLibrary.Load(library.Path)); // loaded once per path
    }

    [Fact]
    public void ATokenizerReportsItsVocabularyAndPadding()
    {
        var tokenizer = Grammar.Require();
        Assert.Equal(264, tokenizer.VocabSize);
        Assert.Equal(9, tokenizer.MaskWords);
        using var padded = Grammar.Library!.CreateTokenizer(Grammar.TokenizerJson, 300, [Grammar.StopId]);
        Assert.Equal(300, padded.VocabSize);
        Assert.Equal(10, padded.MaskWords);
    }

    [Fact]
    public void TokenizerErrorsCarryTheLibraryStatus()
    {
        _ = Grammar.Require();
        var library = Grammar.Library!;
        var small = Assert.Throws<GrammarException>(() => library.CreateTokenizer(Grammar.TokenizerJson, 100, []));
        Assert.Equal(GrammarStatus.Argument, small.Status);
        Assert.Contains("smaller", small.Message, StringComparison.Ordinal);
        var stop = Assert.Throws<GrammarException>(() => library.CreateTokenizer(Grammar.TokenizerJson, 0, [9999]));
        Assert.Equal(GrammarStatus.Argument, stop.Status);
        var junk = Assert.Throws<GrammarException>(() => library.CreateTokenizer("not json"u8, 0, []));
        Assert.Equal(GrammarStatus.Tokenizer, junk.Status);
        Assert.False(junk.IsInvalidGrammar);
    }
}
