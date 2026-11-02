using System.Globalization;
using System.Text.RegularExpressions;

namespace Starlings.Roost.Server.Tests.Support;

/// <summary>One sample of a Prometheus text exposition: name, labels and value.</summary>
internal sealed record PromSample(string Name, IReadOnlyDictionary<string, string> Labels, double Value);

/// <summary>A strict reader of the Prometheus text format (0.0.4) as the server writes it, for the metrics tests.</summary>
internal static partial class Prometheus
{
    public static (List<PromSample> Samples, Dictionary<string, string> Types) Parse(string text)
    {
        var samples = new List<PromSample>();
        var types = new Dictionary<string, string>(StringComparer.Ordinal);
        Assert.EndsWith("\n", text, StringComparison.Ordinal);
        foreach (var line in text.TrimEnd('\n').Split('\n'))
        {
            if (line.StartsWith("# HELP ", StringComparison.Ordinal))
            {
                continue;
            }

            if (line.StartsWith("# TYPE ", StringComparison.Ordinal))
            {
                var parts = line.Split(' ');
                Assert.Equal(4, parts.Length);
                Assert.Matches(NamePattern(), parts[2]);
                Assert.Matches("^(counter|gauge|histogram)$", parts[3]);
                Assert.True(types.TryAdd(parts[2], parts[3]), $"family {parts[2]} declared twice");
                continue;
            }

            var m = SamplePattern().Match(line);
            Assert.True(m.Success, $"not a sample line: {line}");
            var labels = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (Match l in LabelPattern().Matches(m.Groups["labels"].Value))
            {
                labels.Add(l.Groups["k"].Value, l.Groups["v"].Value);
            }

            var value = m.Groups["value"].Value switch
            {
                "+Inf" => double.PositiveInfinity,
                var v => double.Parse(v, NumberStyles.Float, CultureInfo.InvariantCulture),
            };
            samples.Add(new PromSample(m.Groups["name"].Value, labels, value));
        }

        return (samples, types);
    }

    /// <summary>The one sample of <paramref name="name"/> whose labels include <paramref name="labels"/>.</summary>
    public static double Value(List<PromSample> samples, string name, params (string Key, string Value)[] labels) =>
        Assert.Single(samples, s => s.Name == name && labels.All(l => s.Labels.TryGetValue(l.Key, out var v) && v == l.Value)).Value;

    [GeneratedRegex(@"^[a-zA-Z_:][a-zA-Z0-9_:]*$")]
    private static partial Regex NamePattern();

    [GeneratedRegex(@"^(?<name>[a-zA-Z_:][a-zA-Z0-9_:]*)\{(?<labels>[^}]*)\} (?<value>\S+)$")]
    private static partial Regex SamplePattern();

    [GeneratedRegex(@"(?<k>[a-zA-Z_][a-zA-Z0-9_]*)=""(?<v>(?:[^""\\]|\\.)*)""")]
    private static partial Regex LabelPattern();
}
