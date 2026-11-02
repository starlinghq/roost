using System.Text;

namespace Starlings.Roost.Native;

/// <summary>The kinds of grammar <c>roost-grammar</c> compiles (<c>ROOST_GRAMMAR_KIND_*</c>).</summary>
public enum GrammarKind
{
    /// <summary>A JSON schema (the subset llguidance implements; an unimplemented keyword is <see cref="GrammarStatus.Grammar"/>).</summary>
    JsonSchema = 1,

    /// <summary>Any JSON object; the source is ignored.</summary>
    JsonObject = 2,

    /// <summary>A regular expression over the whole output.</summary>
    Regex = 3,

    /// <summary>A Lark grammar in llguidance's dialect.</summary>
    Lark = 4,
}

/// <summary>A grammar to compile: its kind and source text.</summary>
/// <param name="Kind">The kind.</param>
/// <param name="Source">The schema, regular expression or Lark source; empty for <see cref="GrammarKind.JsonObject"/>.</param>
/// <remarks>JSON kinds produce compact JSON (no whitespace between tokens) unless the schema carries llguidance's own
/// <c>x-guidance</c> options.</remarks>
public sealed record GrammarSpec(GrammarKind Kind, string Source)
{
    /// <summary>Any JSON object (OpenAI's <c>response_format</c> <c>json_object</c>).</summary>
    public static GrammarSpec JsonObject { get; } = new(GrammarKind.JsonObject, "");

    /// <summary>A JSON schema (OpenAI's <c>response_format</c> <c>json_schema</c>).</summary>
    /// <param name="schema">The schema as JSON text.</param>
    /// <returns>The spec.</returns>
    public static GrammarSpec JsonSchema(string schema) => new(GrammarKind.JsonSchema, schema);
}

/// <summary>
/// A model's tokenizer environment in <c>roost-grammar</c>: the token trie and slicer tables that every matcher of the model
/// shares. Build one per model (it costs about a second for a 150k vocabulary, S10) and make a matcher per sequence. Making
/// matchers from several threads at once is safe.
/// </summary>
public sealed unsafe class GrammarTokenizer : IDisposable
{
    private readonly RoostGrammarLibrary _library;
    private nint _handle;

    internal GrammarTokenizer(RoostGrammarLibrary library, nint handle)
    {
        _library = library;
        _handle = handle;
        VocabSize = (int)library.TokenizerVocab(handle);
        MaskWords = (int)library.MaskWords(handle);
    }

    /// <summary>Frees the native tokenizer if <see cref="Dispose"/> was not called.</summary>
    ~GrammarTokenizer()
    {
        Free();
    }

    /// <summary>The library.</summary>
    public RoostGrammarLibrary Library => _library;

    /// <summary>Mask rows: the model's logits width given at creation, or the tokenizer's own size.</summary>
    public int VocabSize { get; }

    /// <summary>Words of a mask, <c>ceil(VocabSize / 32)</c>.</summary>
    public int MaskWords { get; }

    /// <summary>Compiles a grammar into a matcher for one sequence.</summary>
    /// <param name="spec">The grammar.</param>
    /// <returns>The matcher.</returns>
    /// <exception cref="GrammarException">The grammar is invalid or unsupported (<see cref="GrammarException.IsInvalidGrammar"/>), or the library failed.</exception>
    public GrammarMatcher CreateMatcher(GrammarSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        var handle = Live();
        var bytes = Encoding.UTF8.GetBytes(spec.Source ?? "");
        nint matcher;
        int rc;
        fixed (byte* data = bytes)
        {
            rc = _library.MatcherNew(handle, (int)spec.Kind, data, (nuint)bytes.Length, &matcher);
        }

        GC.KeepAlive(this);
        _library.Check(rc);
        return new GrammarMatcher(_library, matcher, VocabSize, MaskWords);
    }

    /// <summary>Checks that a grammar compiles, without keeping a matcher.</summary>
    /// <param name="spec">The grammar.</param>
    /// <exception cref="GrammarException">As for <see cref="CreateMatcher"/>.</exception>
    public void Validate(GrammarSpec spec)
    {
        using var matcher = CreateMatcher(spec);
    }

    /// <summary>llguidance's greedy byte tokenization of text (for tests and tools; it need not match the model's tokenizer).</summary>
    /// <param name="text">The text.</param>
    /// <returns>The token ids.</returns>
    public int[] Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var handle = Live();
        var bytes = Encoding.UTF8.GetBytes(text);
        var ids = new uint[bytes.Length + 1];
        nuint count;
        int rc;
        fixed (byte* t = bytes)
        fixed (uint* o = ids)
        {
            rc = _library.Tokenize(handle, t, (nuint)bytes.Length, o, (nuint)ids.Length, &count);
        }

        GC.KeepAlive(this);
        _library.Check(rc);
        var result = new int[(int)count];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = (int)ids[i];
        }

        return result;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Free();
        GC.SuppressFinalize(this);
    }

    private nint Live()
    {
        var handle = Volatile.Read(ref _handle);
        ObjectDisposedException.ThrowIf(handle == 0, this);
        return handle;
    }

    private void Free()
    {
        var handle = Interlocked.Exchange(ref _handle, 0);
        if (handle != 0)
        {
            _library.TokenizerFree(handle);
        }
    }
}
