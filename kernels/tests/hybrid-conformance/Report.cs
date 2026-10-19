namespace Roost.HybridConformance;

/// <summary>Collects one line per check, prints it, and keeps the failure count.</summary>
internal sealed class Report
{
    private readonly List<string> _lines = [];

    public int Failures { get; private set; }

    public int Checks { get; private set; }

    public void Line(string text)
    {
        _lines.Add(text);
        Console.WriteLine(text);
    }

    public void Check(bool ok, string name, string detail)
    {
        Checks++;
        if (!ok)
        {
            Failures++;
        }

        Line($"{(ok ? "PASS" : "FAIL")} {name}: {detail}");
    }

    public void Save(string? path)
    {
        if (path is not null)
        {
            File.WriteAllLines(path, _lines);
        }
    }
}

internal static class Compare
{
    /// <summary>Largest |a - b| and the largest |a - b| / (|b| + floor) over the elements; b is the reference.</summary>
    public static (double MaxAbs, double MaxScaled, long Worst) Errors(Func<long, double> got, Func<long, double> want, long n, double floor)
    {
        double maxAbs = 0;
        double maxScaled = 0;
        long worst = -1;
        for (long i = 0; i < n; i++)
        {
            double g = got(i);
            double w = want(i);
            double d = double.IsNaN(g) || double.IsNaN(w) ? double.PositiveInfinity : Math.Abs(g - w);
            double s = d / (Math.Abs(w) + floor);
            maxAbs = Math.Max(maxAbs, d);
            if (s > maxScaled)
            {
                maxScaled = s;
                worst = i;
            }
        }

        return (maxAbs, maxScaled, worst);
    }
}
