namespace Starlings.Roost.Native;

/// <summary>
/// One grammar's matcher for one sequence: the allowed-token mask for the next token, and the tokens sampled so far. Not thread
/// safe; different matchers may run on different threads at once.
/// </summary>
/// <remarks>
/// After a complete value the matcher stops itself (S10 finding 2): its mask then allows only the model's stop ids, and a stop id
/// passed to <see cref="TryConsume"/> is accepted without changing anything.
/// </remarks>
public sealed unsafe class GrammarMatcher : IDisposable
{
    private readonly RoostGrammarLibrary _library;
    private nint _handle;

    internal GrammarMatcher(RoostGrammarLibrary library, nint handle, int vocabSize, int maskWords)
    {
        _library = library;
        _handle = handle;
        VocabSize = vocabSize;
        MaskWords = maskWords;
    }

    /// <summary>Frees the native matcher if <see cref="Dispose"/> was not called.</summary>
    ~GrammarMatcher()
    {
        Free();
    }

    /// <summary>Mask rows.</summary>
    public int VocabSize { get; }

    /// <summary>Words of a mask.</summary>
    public int MaskWords { get; }

    /// <summary>Whether the output so far is complete (a stop id may follow).</summary>
    /// <exception cref="GrammarException">The matcher is in its error state.</exception>
    public bool IsAccepting
    {
        get
        {
            int rc = _library.IsAccepting(Live());
            GC.KeepAlive(this);
            return rc >= 0 ? rc == 1 : throw Fail(rc);
        }
    }

    /// <summary>Whether the matcher has stopped: the grammar is complete and nothing but a stop id may follow.</summary>
    /// <exception cref="GrammarException">The matcher is in its error state.</exception>
    public bool IsStopped
    {
        get
        {
            int rc = _library.IsStopped(Live());
            GC.KeepAlive(this);
            return rc >= 0 ? rc == 1 : throw Fail(rc);
        }
    }

    /// <summary>Writes the allowed-token mask for the next token: bit <c>v % 32</c> of word <c>v / 32</c> is 1 when token <c>v</c> may follow.</summary>
    /// <param name="mask">Exactly <see cref="MaskWords"/> words.</param>
    /// <exception cref="GrammarException">The matcher is in its error state.</exception>
    public void ComputeMask(Span<uint> mask)
    {
        if (mask.Length != MaskWords)
        {
            throw new ArgumentException($"The mask has {mask.Length} words; the tokenizer needs {MaskWords}.", nameof(mask));
        }

        fixed (uint* m = mask)
        {
            ComputeMask(m);
        }
    }

    /// <summary>Advances the matcher by one sampled token.</summary>
    /// <param name="token">The token.</param>
    /// <returns>True when accepted; false when the grammar does not allow it here (the matcher is unchanged).</returns>
    /// <exception cref="GrammarException">The token is out of range or the matcher is in its error state.</exception>
    public bool TryConsume(int token)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(token);
        int rc = _library.Consume(Live(), (uint)token);
        GC.KeepAlive(this);
        return rc switch
        {
            (int)GrammarStatus.Ok => true,
            (int)GrammarStatus.Rejected => false,
            _ => throw Fail(rc),
        };
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Free();
        GC.SuppressFinalize(this);
    }

    /// <summary>Writes the mask to <paramref name="mask"/>, which holds <see cref="MaskWords"/> words.</summary>
    internal void ComputeMask(uint* mask)
    {
        int rc = _library.ComputeMask(Live(), mask, (nuint)MaskWords);
        GC.KeepAlive(this);
        _library.Check(rc);
    }

    private GrammarException Fail(int rc) => new((GrammarStatus)rc, _library.LastError());

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
            _library.MatcherFree(handle);
        }
    }
}
