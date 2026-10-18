using System.Diagnostics;

namespace S1;

internal static class Stats
{
    public static double NsPerTick => 1e9 / Stopwatch.Frequency;

    public static string Summarize(string name, long[] ticks, double baselineNs = 0)
    {
        Array.Sort(ticks);
        double p50 = ticks[ticks.Length / 2] * NsPerTick;
        double p99 = ticks[(int)(ticks.Length * 0.99)] * NsPerTick;
        double max = ticks[^1] * NsPerTick;
        string adj = baselineNs > 0 ? $" | minus timer baseline {baselineNs:F0} ns: p50={Math.Max(0, p50 - baselineNs):F0} p99={Math.Max(0, p99 - baselineNs):F0}" : "";
        return $"{name,-34} n={ticks.Length,8} p50={p50,10:F0} ns p99={p99,10:F0} ns max={max,12:F0} ns{adj}";
    }
}

internal static class Rss
{
    public static string Status()
    {
        long rss = 0, anon = 0, file = 0;
        foreach (var l in File.ReadLines("/proc/self/status"))
        {
            if (l.StartsWith("VmRSS:")) rss = Kb(l);
            else if (l.StartsWith("RssAnon:")) anon = Kb(l);
            else if (l.StartsWith("RssFile:")) file = Kb(l);
        }
        var gc = GC.GetGCMemoryInfo();
        return $"VmRSS={rss / 1024.0:F1} MiB (anon {anon / 1024.0:F1}, file {file / 1024.0:F1}) gc_heap={GC.GetTotalMemory(false) / 1048576.0:F1} MiB gc_committed={gc.TotalCommittedBytes / 1048576.0:F1} MiB";
    }

    static long Kb(string l) => long.Parse(l.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1]);

    // Rss per mapping group from smaps: libcuda, libcudart, other libcuda*, our .so, other shared libs, the executable, anonymous/heap.
    public static string Smaps()
    {
        var groups = new SortedDictionary<string, long>();
        string cur = "[anon]";
        foreach (var l in File.ReadLines("/proc/self/smaps"))
        {
            if (l.Length > 0 && (char.IsAsciiHexDigit(l[0])) && l.Contains('-') && l.IndexOf(' ') > 8 && !l.StartsWith("Rss:"))
            {
                var parts = l.Split(' ', 6, StringSplitOptions.RemoveEmptyEntries);
                string path = parts.Length >= 6 ? parts[5].Trim() : "[anon]";
                cur = Classify(path);
            }
            else if (l.StartsWith("Rss:")) { groups[cur] = groups.GetValueOrDefault(cur) + Kb(l); }
        }
        return string.Join("\n", groups.OrderByDescending(g => g.Value).Select(g => $"    {g.Value / 1024.0,8:F1} MiB  {g.Key}"));
    }

    static string Classify(string p)
    {
        if (p.Length == 0) return "[anon]";
        if (p.StartsWith('[')) return p;
        string f = Path.GetFileName(p);
        if (f.StartsWith("libcuda.so")) return "libcuda (driver)";
        if (f.StartsWith("libcudart")) return "libcudart";
        if (f.StartsWith("libs1kernels")) return "libs1kernels (kernel images)";
        if (f.StartsWith("libnvidia") || f.StartsWith("libnvcuvid")) return "nvidia libs: " + f;
        if (f.StartsWith("s1app")) return "s1app (AOT image)";
        if (f.Contains(".so")) return "other shared libs";
        return "other files (" + f + ")";
    }
}
