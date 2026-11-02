using System.Globalization;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Onboarding;

/// <summary>The <c>roost-onboard</c> command line.</summary>
internal static class Program
{
    internal const string Usage = """
        roost-onboard inspect <config.json> [--block-size N] [--kv-dtype BF16|F16|F32|F8E4M3|F8E5M2] [--snapshot-slots N] [--out report.json]
            Maps a Hugging Face config.json to the C3 op vocabulary and C4 state groups. Writes the JSON report to stdout (or --out).
            Exit 0: composes. 1: composes_pending_reference or missing_ops. 2: the config cannot be read, or bad arguments.
        roost-onboard validate-recipe [--root <repository root>] <recipe.json | directory>...
            Validates roost-v1 recipes against the strict schema and the rules beyond it. One "path: location: message" line per problem.
            Exit 0: every recipe is valid. 1: problems. 2: bad arguments.
        roost-onboard schema
            Writes the roost-v1 recipe JSON Schema to stdout.
        """;

    private static int Main(string[] args) => Run(args, Console.Out, Console.Error);

    /// <summary>Runs one command.</summary>
    /// <param name="args">Arguments.</param>
    /// <param name="stdout">Standard output.</param>
    /// <param name="stderr">Standard error.</param>
    /// <returns>The exit code.</returns>
    internal static int Run(string[] args, TextWriter stdout, TextWriter stderr)
    {
        if (args.Length == 0)
        {
            stderr.WriteLine(Usage);
            return 2;
        }

        try
        {
            return args[0] switch
            {
                "inspect" => Inspect(args[1..], stdout, stderr),
                "validate-recipe" => ValidateRecipes(args[1..], stdout, stderr),
                "schema" when args.Length == 1 => WriteSchema(stdout),
                _ => UsageError(stderr, $"unknown command '{args[0]}'"),
            };
        }
        catch (ArgumentException e)
        {
            return UsageError(stderr, e.Message);
        }
    }

    private static int Inspect(string[] args, TextWriter stdout, TextWriter stderr)
    {
        string? config = null;
        string? output = null;
        var options = new InspectOptions();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--block-size":
                    options = options with { BlockSize = PositiveInt(args, ref i) };
                    break;
                case "--snapshot-slots":
                    options = options with { SnapshotSlots = PositiveInt(args, ref i, allowZero: true) };
                    break;
                case "--kv-dtype":
                    string name = Value(args, ref i);
                    options = options with
                    {
                        KvDType = name is "BF16" or "F16" or "F32" or "F8E4M3" or "F8E5M2"
                            ? Enum.Parse<DType>(name)
                            : throw new ArgumentException($"--kv-dtype '{name}' is not BF16, F16, F32, F8E4M3 or F8E5M2."),
                    };
                    break;
                case "--out":
                    output = Value(args, ref i);
                    break;
                default:
                    if (args[i].StartsWith("--", StringComparison.Ordinal) || config is not null)
                    {
                        throw new ArgumentException($"unexpected argument '{args[i]}'");
                    }

                    config = args[i];
                    break;
            }
        }

        if (config is null)
        {
            throw new ArgumentException("inspect needs a config.json path");
        }

        InspectReport report;
        try
        {
            report = ConfigInspector.InspectFile(config, options);
        }
        catch (Exception e) when (e is OnboardingConfigException or IOException or UnauthorizedAccessException)
        {
            stderr.WriteLine($"{config}: {e.Message}");
            return 2;
        }

        string json = ConfigInspector.ToJson(report);
        if (output is null)
        {
            stdout.WriteLine(json);
        }
        else
        {
            File.WriteAllText(output, json + Environment.NewLine);
        }

        stderr.WriteLine($"{config}: {report.Verdict switch
        {
            CompositionVerdict.Composes => "composes",
            CompositionVerdict.ComposesPendingReference => "composes_pending_reference: " + string.Join(", ", report.PendingOps),
            _ => "missing_ops: " + string.Join(", ", report.MissingOps),
        }}");
        return report.Verdict == CompositionVerdict.Composes ? 0 : 1;
    }

    private static int ValidateRecipes(string[] args, TextWriter stdout, TextWriter stderr)
    {
        string? root = null;
        var paths = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--root")
            {
                root = Value(args, ref i);
            }
            else if (args[i].StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"unexpected argument '{args[i]}'");
            }
            else
            {
                paths.Add(args[i]);
            }
        }

        if (paths.Count == 0)
        {
            throw new ArgumentException("validate-recipe needs at least one recipe or directory");
        }

        var errors = new RecipeValidator(root).ValidatePaths(paths);
        foreach (string e in errors)
        {
            stdout.WriteLine(e);
        }

        if (errors.Count == 0)
        {
            stderr.WriteLine("every recipe is valid");
        }

        return errors.Count == 0 ? 0 : 1;
    }

    private static int WriteSchema(TextWriter stdout)
    {
        stdout.Write(RecipeValidator.SchemaText());
        return 0;
    }

    private static int UsageError(TextWriter stderr, string message)
    {
        stderr.WriteLine("roost-onboard: " + message);
        stderr.WriteLine(Usage);
        return 2;
    }

    private static string Value(string[] args, ref int i) =>
        ++i < args.Length ? args[i] : throw new ArgumentException($"{args[i - 1]} needs a value");

    private static int PositiveInt(string[] args, ref int i, bool allowZero = false)
    {
        string flag = args[i];
        string v = Value(args, ref i);
        return int.TryParse(v, NumberStyles.None, CultureInfo.InvariantCulture, out int n) && (n > 0 || (allowZero && n == 0))
            ? n
            : throw new ArgumentException($"{flag} '{v}' is not a {(allowZero ? "non-negative" : "positive")} integer");
    }
}
