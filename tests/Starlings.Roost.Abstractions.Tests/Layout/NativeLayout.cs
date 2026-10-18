using System.Globalization;
using System.Reflection;
using System.Text.RegularExpressions;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Tests.Layout;

/// <summary>The golden layout file and the C headers, parsed. Built by tools/abi-layout.py with gcc.</summary>
internal static partial class NativeLayout
{
    private static readonly string s_dir = Path.Combine(AppContext.BaseDirectory, "Layout");

    public static IReadOnlyDictionary<string, NativeStruct> Golden { get; } = ParseGolden(
        File.ReadAllLines(Path.Combine(s_dir, "abi-layout.golden.txt")));

    public static IReadOnlyDictionary<string, long> Defines { get; } = ParseDefines();

    public static IReadOnlyList<Type> MirroredTypes { get; } =
    [
        .. typeof(ContractVersion).Assembly.GetTypes()
            .Where(t => t.GetCustomAttribute<NativeStructAttribute>() is not null)
            .OrderBy(t => t.FullName, StringComparer.Ordinal),
    ];

    public static string NativeName(Type type) => type.GetCustomAttribute<NativeStructAttribute>()!.Name;

    /// <summary>snake_case to PascalCase: <c>q_indptr_host</c> to <c>QIndptrHost</c>.</summary>
    public static string ToPascal(string snake) =>
        string.Concat(snake.Split('_').Select(p => p.Length == 0 ? p : char.ToUpperInvariant(p[0]) + p[1..]));

    private static Dictionary<string, NativeStruct> ParseGolden(string[] lines)
    {
        var result = new Dictionary<string, NativeStruct>(StringComparer.Ordinal);
        NativeStruct? current = null;
        foreach (var raw in lines)
        {
            // Trim: a Windows checkout may carry CRLF.
            var line = raw.Trim();
            if (line.Length == 0 || line.StartsWith('#'))
            {
                continue;
            }

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts[0] == "struct")
            {
                current = new NativeStruct(parts[1], Value(parts[2]), Value(parts[3]), []);
                result.Add(current.Name, current);
            }
            else
            {
                current!.Fields.Add(new NativeField(parts[0], Int(parts[1]), Int(parts[2])));
            }
        }

        return result;

        static int Value(string pair) => Int(pair[(pair.IndexOf('=', StringComparison.Ordinal) + 1)..]);
        static int Int(string s) => int.Parse(s, CultureInfo.InvariantCulture);
    }

    private static Dictionary<string, long> ParseDefines()
    {
        var result = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var header in Directory.GetFiles(Path.Combine(s_dir, "include"), "*.h"))
        {
            foreach (var raw in File.ReadAllLines(header))
            {
                var match = DefineLine().Match(raw.Trim());
                if (match.Success)
                {
                    var text = match.Groups["value"].Value;
                    var value = text.StartsWith("0x", StringComparison.OrdinalIgnoreCase)
                        ? long.Parse(text[2..], NumberStyles.HexNumber, CultureInfo.InvariantCulture)
                        : long.Parse(text, CultureInfo.InvariantCulture);
                    result.Add(match.Groups["name"].Value, value);
                }
            }
        }

        return result;
    }

    [GeneratedRegex(@"^#define (?<name>ROOST_\w+) (?<value>0x[0-9A-Fa-f]+|\d+)u?\b")]
    private static partial Regex DefineLine();
}

internal sealed record NativeStruct(string Name, int Size, int Align, List<NativeField> Fields);

internal sealed record NativeField(string Name, int Offset, int Size);
