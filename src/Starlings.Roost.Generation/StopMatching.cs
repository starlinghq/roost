using Starlings.Generation.OnnxGenAI;

namespace Starlings.Roost.Generation;

/// <summary>Result of pushing text into an <see cref="IStopMatcher"/>.</summary>
/// <param name="Emit">Text safe to stream: it cannot be part of a stop sequence.</param>
/// <param name="Matched">The stop sequence that completed, or null; text after it is dropped.</param>
public readonly record struct StopMatchResult(string Emit, string? Matched);

/// <summary>
/// The seam the engine calls for stop sequences. Text that could still become a stop sequence is held back until it completes a
/// match or cannot. <see cref="StarlingStopMatcherFactory"/> implements it over Starling's published <c>StopSequenceMatcher</c>,
/// so stop strings behave as they do in Starling's own <c>TextGenerator</c>.
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
/// Stop matching through Starling's public <see cref="StopSequenceMatcher"/> (public since <c>Starlings.Generation.OnnxGenAI</c>
/// 0.3.0, starling#124): ordinal comparison, partial matches held back, the earliest match wins and at the same start the longest.
/// </summary>
public sealed class StarlingStopMatcherFactory : IStopMatcherFactory
{
    /// <summary>The shared instance.</summary>
    public static StarlingStopMatcherFactory Instance { get; } = new();

    /// <inheritdoc />
    /// <exception cref="ArgumentException">A stop sequence is null or empty.</exception>
    public IStopMatcher Create(IReadOnlyList<string> stopSequences)
    {
        ArgumentNullException.ThrowIfNull(stopSequences);
        return new StarlingStopMatcher(new StopSequenceMatcher(stopSequences));
    }

    private sealed class StarlingStopMatcher(StopSequenceMatcher inner) : IStopMatcher
    {
        public StopMatchResult Push(string text)
        {
            var (emit, matched) = inner.Push(text);
            return new StopMatchResult(emit, matched);
        }

        public string Flush() => inner.Flush();
    }
}
