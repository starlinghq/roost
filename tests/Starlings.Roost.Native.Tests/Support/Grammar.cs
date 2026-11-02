using Starlings.Roost.Native;

namespace Starlings.Roost.Native.Tests.Support;

/// <summary>
/// The locally built roost-grammar library and the tiny tokenizer over it. The library is a release asset and is not built by
/// <c>dotnet test</c>: tests that need it skip with <see cref="Reason"/> when it is absent, the way the kernel tests skip without a
/// kernel library. Looked up in <c>ROOST_GRAMMAR_LIBRARY</c>, then <c>native/roost-grammar/target/{release,debug}</c>.
/// </summary>
internal static class Grammar
{
    public const int StopId = 263;
    public const int TokenizerVocab = 264;

    private static readonly Lazy<(RoostGrammarLibrary? Library, string Reason)> s_library = new(Find);
    private static readonly Lazy<GrammarTokenizer?> s_tokenizer = new(() =>
        Library?.CreateTokenizer(TokenizerJson, 0, [StopId]));

    public static RoostGrammarLibrary? Library => s_library.Value.Library;

    public static string Reason => s_library.Value.Reason;

    /// <summary>The tiny tokenizer (264 rows, stop id 263), shared by the tests; null when the library is absent.</summary>
    public static GrammarTokenizer? Tokenizer => s_tokenizer.Value;

    public static byte[] TokenizerJson => File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "tiny-bytelevel-bpe.tokenizer.json"));

    /// <summary>Skips the test when the library is not built.</summary>
    public static GrammarTokenizer Require()
    {
        Assert.SkipUnless(Library is not null, Reason);
        return Tokenizer!;
    }

    public static bool Allowed(ReadOnlySpan<uint> mask, int token) => ((mask[token >> 5] >> (token & 31)) & 1u) == 1u;

    public static int[] AllowedTokens(ReadOnlySpan<uint> mask, int vocab)
    {
        var list = new List<int>();
        for (var v = 0; v < vocab; v++)
        {
            if (Allowed(mask, v))
            {
                list.Add(v);
            }
        }

        return [.. list];
    }

    /// <summary>The repository's built library path, or null.</summary>
    public static string? RepositoryBuild()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Roost.slnx")))
            {
                foreach (var profile in new[] { "release", "debug" })
                {
                    var path = Path.Combine(dir.FullName, "native", "roost-grammar", "target", profile, RoostGrammarLibrary.FileName);
                    if (File.Exists(path))
                    {
                        return path;
                    }
                }

                return null;
            }
        }

        return null;
    }

    private static (RoostGrammarLibrary?, string) Find()
    {
        var configured = Environment.GetEnvironmentVariable(RoostGrammarLibrary.PathVariable);
        var path = !string.IsNullOrEmpty(configured) ? configured : RepositoryBuild();
        if (path is null)
        {
            return (null, $"roost-grammar is not built: run `cargo build --release --locked` in native/roost-grammar, or set {RoostGrammarLibrary.PathVariable} to the library file");
        }

        try
        {
            return (RoostGrammarLibrary.Load(path), "");
        }
        catch (GrammarException e)
        {
            return (null, e.Message);
        }
    }
}
