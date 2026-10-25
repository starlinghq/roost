using Starlings.Roost.Generation;

namespace Starlings.Generation.Roost.Tests.Support;

/// <summary>
/// A plain stop matcher for tests of the client's use of the <see cref="IStopMatcherFactory"/> seam (the Starling-backed
/// factory lands separately): holds back the longest suffix that could start a stop string and truncates at a match.
/// </summary>
internal sealed class SimpleStopMatcherFactory : IStopMatcherFactory
{
    public List<IReadOnlyList<string>> Created { get; } = [];

    public IStopMatcher Create(IReadOnlyList<string> stopSequences)
    {
        Created.Add(stopSequences);
        return new Matcher(stopSequences);
    }

    private sealed class Matcher(IReadOnlyList<string> stops) : IStopMatcher
    {
        private string _held = string.Empty;

        public StopMatchResult Push(string text)
        {
            var buffer = _held + text;
            var best = -1;
            string? matched = null;
            foreach (var s in stops)
            {
                var at = buffer.IndexOf(s, StringComparison.Ordinal);
                if (at >= 0 && (best < 0 || at < best))
                {
                    best = at;
                    matched = s;
                }
            }

            if (matched is not null)
            {
                _held = string.Empty;
                return new StopMatchResult(buffer[..best], matched);
            }

            var keep = 0;
            foreach (var s in stops)
            {
                for (var n = Math.Min(s.Length - 1, buffer.Length); n > keep; n--)
                {
                    if (buffer.EndsWith(s[..n], StringComparison.Ordinal))
                    {
                        keep = n;
                        break;
                    }
                }
            }

            _held = buffer[^keep..];
            return new StopMatchResult(buffer[..^keep], null);
        }

        public string Flush()
        {
            var held = _held;
            _held = string.Empty;
            return held;
        }
    }
}
