namespace Starlings.Roost.Models.Qwen35.Tests;

/// <summary>The committed tensor listing of the locked checkpoint's safetensors header (text and MTP tensors): name, dtype, shape.</summary>
internal static class CheckpointHeader
{
    internal sealed record Tensor(string Name, string Dtype, long[] Shape)
    {
        public long Elements => Shape.Aggregate(1L, (a, d) => a * d);
    }

    internal static IReadOnlyDictionary<string, Tensor> Tensors { get; } = Read();

    private static Dictionary<string, Tensor> Read()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Data", "qwen35-0.8b.safetensors-header.txt");
        var result = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (line.Length == 0 || line[0] == '#')
            {
                continue;
            }

            string[] f = line.Split(' ');
            result.Add(f[0], new Tensor(f[0], f[1], f[2].Split(',').Select(x => long.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray()));
        }

        return result;
    }
}
