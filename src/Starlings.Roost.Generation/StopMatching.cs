namespace Starlings.Roost.Generation;

/// <summary>Result of pushing text into an <see cref="IStopMatcher"/>.</summary>
/// <param name="Emit">Text safe to stream: it cannot be part of a stop sequence.</param>
/// <param name="Matched">The stop sequence that completed, or null; text after it is dropped.</param>
public readonly record struct StopMatchResult(string Emit, string? Matched);

/// <summary>
/// The seam the engine calls for stop sequences. Text that could still become a stop sequence is held back until it completes a
/// match or cannot. This shape is Roost's own design. It is not verified against Starling's internal <c>StopSequenceMatcher</c>
/// (hold-back and truncation behaviour are unchecked, and only the empty-list pass-through is tested); the Starling-backed
/// implementation and its equivalence tests arrive with starling#124.
/// </summary>
public interface IStopMatcher
{
    /// <summary>Adds streamed text.</summary>
    /// <param name="text">The next piece of text.</param>
    /// <returns>The text to emit and the matched stop sequence, if any.</returns>
    StopMatchResult Push(string text);

    /// <summary>Returns the held-back text (no further text will arrive) and clears it.</summary>
    /// <returns>The held-back text.</returns>
    string Flush();
}

/// <summary>Creates one <see cref="IStopMatcher"/> per request.</summary>
public interface IStopMatcherFactory
{
    /// <summary>Creates a matcher.</summary>
    /// <param name="stopSequences">The request's stop strings; empty for none.</param>
    /// <returns>A new matcher.</returns>
    IStopMatcher Create(IReadOnlyList<string> stopSequences);
}

/// <summary>
/// The factory available until Starling publishes a stop-matching API (starling#124, no version of
/// <c>Starlings.Generation.OnnxGenAI</c> on nuget.org exposes one). Without stop strings it returns a pass-through matcher; with
/// stop strings it throws rather than approximate Starling's matcher.
/// </summary>
public sealed class PendingStopMatcherFactory : IStopMatcherFactory
{
    /// <summary>The shared instance.</summary>
    public static PendingStopMatcherFactory Instance { get; } = new();

    /// <inheritdoc />
    /// <exception cref="NotSupportedException"><paramref name="stopSequences"/> is not empty.</exception>
    public IStopMatcher Create(IReadOnlyList<string> stopSequences)
    {
        ArgumentNullException.ThrowIfNull(stopSequences);
        if (stopSequences.Count > 0)
        {
            throw new NotSupportedException(
                "Stop strings need Starling's public stop-matching API (starling#124), which no published Starlings.Generation.OnnxGenAI version exposes yet.");
        }

        return new PassThroughStopMatcher();
    }

    private sealed class PassThroughStopMatcher : IStopMatcher
    {
        public StopMatchResult Push(string text)
        {
            ArgumentNullException.ThrowIfNull(text);
            return new StopMatchResult(text, null);
        }

        public string Flush() => string.Empty;
    }
}
