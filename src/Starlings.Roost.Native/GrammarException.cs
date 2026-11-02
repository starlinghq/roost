namespace Starlings.Roost.Native;

/// <summary>The status codes of <c>native/roost-grammar/include/roost_grammar.h</c> (<c>ROOST_GRAMMAR_E_*</c>).</summary>
public enum GrammarStatus
{
    /// <summary><c>ROOST_GRAMMAR_OK</c>.</summary>
    Ok = 0,

    /// <summary>A null pointer, a size mismatch, an unknown kind, a token out of range or text that is not UTF-8.</summary>
    Argument = -1,

    /// <summary>The grammar or schema is invalid or uses something the matcher does not implement: the caller's input.</summary>
    Grammar = -2,

    /// <summary>The tokenizer description could not be read.</summary>
    Tokenizer = -3,

    /// <summary>The token is not allowed in the matcher's state; the matcher is unchanged.</summary>
    Rejected = -4,

    /// <summary>The matcher is in its error state and can only be disposed.</summary>
    State = -5,

    /// <summary>A panic was caught at the library boundary.</summary>
    Panic = -6,
}

/// <summary>A failure reported by <c>roost-grammar</c>, or a library that cannot be loaded.</summary>
public sealed class GrammarException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GrammarException()
        : this(GrammarStatus.Argument, "roost-grammar failed.")
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">The message.</param>
    public GrammarException(string message)
        : this(GrammarStatus.Argument, message)
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">The message.</param>
    /// <param name="innerException">The cause.</param>
    public GrammarException(string message, Exception innerException)
        : base(message, innerException)
    {
        Status = GrammarStatus.Argument;
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="status">The library's status.</param>
    /// <param name="message">The library's message (never request text beyond what the grammar compiler quotes).</param>
    public GrammarException(GrammarStatus status, string message)
        : base(message)
    {
        Status = status;
    }

    /// <summary>The library's status.</summary>
    public GrammarStatus Status { get; }

    /// <summary>Whether the request's own grammar or schema is at fault (an HTTP 400, S10 finding 3), not Roost.</summary>
    public bool IsInvalidGrammar => Status == GrammarStatus.Grammar;
}
