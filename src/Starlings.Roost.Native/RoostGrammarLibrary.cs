using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Text;

namespace Starlings.Roost.Native;

/// <summary>
/// The loaded <c>roost-grammar</c> library (<c>native/roost-grammar</c>, R-051): llguidance token masks behind the C ABI of
/// <c>roost_grammar.h</c>, called through function pointers resolved once at load. A library is loaded once per path and stays
/// loaded for the life of the process, since tokenizers and matchers made from it may outlive any one owner.
/// </summary>
/// <remarks>
/// <para>The library is a release asset like the kernel library (decision 22): it is never committed, and a development build
/// runs <c>cargo build --release --locked</c> in <c>native/roost-grammar</c>. <see cref="TryLoad(out RoostGrammarLibrary?, out string)"/>
/// looks in <see cref="PathVariable"/>, then beside this assembly, then in the platform's default search.</para>
/// <para>ABI rule: the library's major version must equal <see cref="AbiMajor"/> and its minor must be at least
/// <see cref="AbiMinor"/>.</para>
/// </remarks>
public sealed unsafe class RoostGrammarLibrary
{
    /// <summary>The C ABI major version these bindings call.</summary>
    public const int AbiMajor = 1;

    /// <summary>The lowest C ABI minor version these bindings need.</summary>
    public const int AbiMinor = 0;

    /// <summary>The environment variable naming the library file to load.</summary>
    public const string PathVariable = "ROOST_GRAMMAR_LIBRARY";

    private static readonly ConcurrentDictionary<string, RoostGrammarLibrary> s_loaded = new(StringComparer.Ordinal);

    private RoostGrammarLibrary(string path, nint handle)
    {
        Path = path;
        AbiVersionFn = (delegate* unmanaged<uint>)Export(handle, "roost_grammar_abi_version");
        LastErrorFn = (delegate* unmanaged<byte*, nuint, nuint>)Export(handle, "roost_grammar_last_error");
        TokenizerNew = (delegate* unmanaged<byte*, nuint, uint, uint*, nuint, nint*, int>)Export(handle, "roost_grammar_tokenizer_new");
        TokenizerFree = (delegate* unmanaged<nint, void>)Export(handle, "roost_grammar_tokenizer_free");
        TokenizerVocab = (delegate* unmanaged<nint, uint>)Export(handle, "roost_grammar_tokenizer_vocab");
        MaskWords = (delegate* unmanaged<nint, nuint>)Export(handle, "roost_grammar_mask_words");
        Tokenize = (delegate* unmanaged<nint, byte*, nuint, uint*, nuint, nuint*, int>)Export(handle, "roost_grammar_tokenize");
        MatcherNew = (delegate* unmanaged<nint, int, byte*, nuint, nint*, int>)Export(handle, "roost_grammar_matcher_new");
        MatcherFree = (delegate* unmanaged<nint, void>)Export(handle, "roost_grammar_matcher_free");
        ComputeMask = (delegate* unmanaged<nint, uint*, nuint, int>)Export(handle, "roost_grammar_matcher_compute_mask");
        Consume = (delegate* unmanaged<nint, uint, int>)Export(handle, "roost_grammar_matcher_consume");
        IsAccepting = (delegate* unmanaged<nint, int>)Export(handle, "roost_grammar_matcher_is_accepting");
        IsStopped = (delegate* unmanaged<nint, int>)Export(handle, "roost_grammar_matcher_is_stopped");
        AbiVersion = AbiVersionFn();
    }

    /// <summary>The file this library was loaded from (or the bare name the default search resolved).</summary>
    public string Path { get; }

    /// <summary>The library's C ABI version, <c>(major &lt;&lt; 16) | minor</c>.</summary>
    public uint AbiVersion { get; }

    internal delegate* unmanaged<uint> AbiVersionFn { get; }

    internal delegate* unmanaged<byte*, nuint, nuint> LastErrorFn { get; }

    internal delegate* unmanaged<byte*, nuint, uint, uint*, nuint, nint*, int> TokenizerNew { get; }

    internal delegate* unmanaged<nint, void> TokenizerFree { get; }

    internal delegate* unmanaged<nint, uint> TokenizerVocab { get; }

    internal delegate* unmanaged<nint, nuint> MaskWords { get; }

    internal delegate* unmanaged<nint, byte*, nuint, uint*, nuint, nuint*, int> Tokenize { get; }

    internal delegate* unmanaged<nint, int, byte*, nuint, nint*, int> MatcherNew { get; }

    internal delegate* unmanaged<nint, void> MatcherFree { get; }

    internal delegate* unmanaged<nint, uint*, nuint, int> ComputeMask { get; }

    internal delegate* unmanaged<nint, uint, int> Consume { get; }

    internal delegate* unmanaged<nint, int> IsAccepting { get; }

    internal delegate* unmanaged<nint, int> IsStopped { get; }

    /// <summary>The library's file name on the current platform: <c>libroost_grammar.so</c>, <c>roost_grammar.dll</c> or <c>libroost_grammar.dylib</c>.</summary>
    public static string FileName =>
        OperatingSystem.IsWindows() ? "roost_grammar.dll" : OperatingSystem.IsMacOS() ? "libroost_grammar.dylib" : "libroost_grammar.so";

    /// <summary>Loads the library from a file (once per full path) and checks its ABI version.</summary>
    /// <param name="path">The library file.</param>
    /// <returns>The library.</returns>
    /// <exception cref="GrammarException">The file cannot be loaded, lacks an export, or has an incompatible ABI version.</exception>
    public static RoostGrammarLibrary Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        var full = System.IO.Path.GetFullPath(path);
        if (s_loaded.TryGetValue(full, out var loaded))
        {
            return loaded;
        }

        nint handle;
        try
        {
            handle = NativeLibrary.Load(full);
        }
        catch (Exception e) when (e is DllNotFoundException or BadImageFormatException)
        {
            throw new GrammarException($"roost-grammar could not be loaded from {full}: {e.Message}", e);
        }

        return s_loaded.GetOrAdd(full, _ => Checked(new RoostGrammarLibrary(full, handle)));
    }

    /// <summary>
    /// Loads the library from <see cref="PathVariable"/> when it is set, else from this assembly's directory, else by the platform's
    /// default search for <see cref="FileName"/>.
    /// </summary>
    /// <param name="library">The library, or null.</param>
    /// <param name="reason">Why no library was loaded; empty on success.</param>
    /// <returns>Whether a library was loaded.</returns>
    public static bool TryLoad(out RoostGrammarLibrary? library, out string reason)
    {
        library = null;
        try
        {
            var configured = Environment.GetEnvironmentVariable(PathVariable);
            if (!string.IsNullOrEmpty(configured))
            {
                library = Load(configured);
                reason = "";
                return true;
            }

            var beside = System.IO.Path.Combine(AppContext.BaseDirectory, FileName);
            if (File.Exists(beside))
            {
                library = Load(beside);
                reason = "";
                return true;
            }

            if (NativeLibrary.TryLoad(FileName, typeof(RoostGrammarLibrary).Assembly, null, out var handle))
            {
                library = s_loaded.GetOrAdd(FileName, _ => Checked(new RoostGrammarLibrary(FileName, handle)));
                reason = "";
                return true;
            }

            reason = $"{FileName} was not found ({PathVariable} is not set, the file is not beside the application, and the default search failed); "
                + "build it with `cargo build --release --locked` in native/roost-grammar";
            return false;
        }
        catch (GrammarException e)
        {
            reason = e.Message;
            return false;
        }
    }

    /// <summary>Builds the tokenizer environment that every matcher of one model shares (about a second for a 150k vocabulary, S10).</summary>
    /// <param name="tokenizerJson">The model's <c>tokenizer.json</c> bytes.</param>
    /// <param name="vocabSize">The model's logits width; masks have this many rows. 0 keeps the tokenizer's own size.</param>
    /// <param name="stopTokenIds">The model's generation stop ids; they replace the tokenizer's EOS when not empty (S10 finding 1).</param>
    /// <returns>The tokenizer; dispose it when no new matcher is needed (live matchers stay valid).</returns>
    /// <exception cref="GrammarException">The description cannot be read, the vocabulary is smaller than the tokenizer's, or a stop id is out of range.</exception>
    public GrammarTokenizer CreateTokenizer(ReadOnlySpan<byte> tokenizerJson, int vocabSize, ReadOnlySpan<int> stopTokenIds)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(vocabSize);
        var stops = new uint[stopTokenIds.Length];
        for (var i = 0; i < stops.Length; i++)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(stopTokenIds[i], nameof(stopTokenIds));
            stops[i] = (uint)stopTokenIds[i];
        }

        nint handle;
        int rc;
        fixed (byte* json = tokenizerJson)
        fixed (uint* s = stops)
        {
            rc = TokenizerNew(json, (nuint)tokenizerJson.Length, (uint)vocabSize, s, (nuint)stops.Length, &handle);
        }

        Check(rc);
        return new GrammarTokenizer(this, handle);
    }

    /// <summary>Throws the last error of this thread when <paramref name="rc"/> is not <see cref="GrammarStatus.Ok"/>.</summary>
    internal void Check(int rc)
    {
        if (rc != (int)GrammarStatus.Ok)
        {
            throw new GrammarException((GrammarStatus)rc, LastError());
        }
    }

    /// <summary>The calling thread's last error message.</summary>
    internal string LastError()
    {
        Span<byte> buffer = stackalloc byte[1024];
        nuint length;
        fixed (byte* b = buffer)
        {
            length = LastErrorFn(b, (nuint)buffer.Length);
        }

        if (length < (nuint)buffer.Length)
        {
            return Encoding.UTF8.GetString(buffer[..(int)length]);
        }

        var large = new byte[(int)length + 1];
        fixed (byte* b = large)
        {
            length = LastErrorFn(b, (nuint)large.Length);
        }

        return Encoding.UTF8.GetString(large, 0, (int)Math.Min(length, (nuint)(large.Length - 1)));
    }

    private static RoostGrammarLibrary Checked(RoostGrammarLibrary library)
    {
        int major = (int)(library.AbiVersion >> 16);
        int minor = (int)(library.AbiVersion & 0xFFFF);
        if (major != AbiMajor || minor < AbiMinor)
        {
            throw new GrammarException($"roost-grammar at {library.Path} has C ABI {major}.{minor}; these bindings need {AbiMajor}.{AbiMinor} or a later minor.");
        }

        return library;
    }

    private static nint Export(nint handle, string name) =>
        NativeLibrary.TryGetExport(handle, name, out var address)
            ? address
            : throw new GrammarException($"roost-grammar lacks the export {name}; it is not a roost-grammar library or is too old.");
}
