using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

namespace Starlings.Roost.Generation;

/// <summary>
/// Turns generated token ids into text pieces without splitting a character. Delegates to Starling's
/// <see cref="IncrementalDetokenizer"/>: the concatenation of all pieces and the flush equals the tokenizer's decode of the ids.
/// One instance per request; not thread-safe.
/// </summary>
public sealed class RoostIncrementalDetokenizer
{
    private readonly IncrementalDetokenizer _inner;

    /// <summary>Creates a detokenizer.</summary>
    /// <param name="tokenizer">The model's tokenizer.</param>
    /// <param name="skipSpecialTokens">Whether special tokens are dropped from the text.</param>
    public RoostIncrementalDetokenizer(Tokenizer tokenizer, bool skipSpecialTokens = true)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        _inner = new IncrementalDetokenizer(tokenizer, skipSpecialTokens);
    }

    /// <summary>Gets the number of ids pushed since creation or the last <see cref="Flush"/>.</summary>
    public int Count => _inner.Count;

    /// <summary>Adds one id.</summary>
    /// <param name="tokenId">The generated token id.</param>
    /// <returns>The text that became complete, possibly empty.</returns>
    public string Push(int tokenId) => _inner.Push(tokenId);

    /// <summary>Returns text still held back (an incomplete sequence decodes with U+FFFD) and resets the state.</summary>
    /// <returns>The remaining text.</returns>
    public string Flush() => _inner.Flush();
}
