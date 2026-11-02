using System.Globalization;
using System.Text.Json;

namespace Starlings.Roost.Onboarding;

/// <summary>
/// Validates roost-v1 recipes: first against the strict schema (<c>Schemas/roost-recipe-v1.schema.json</c>, embedded), then
/// against the rules a schema cannot express.
/// </summary>
/// <remarks>
/// <para>Rules beyond the schema:</para>
/// <list type="bullet">
/// <item>A recipe file is named <c>&lt;id&gt;.json</c>, and ids are unique across the files validated together.</item>
/// <item>A status other than <c>unsupported</c> names evidence; <c>supported</c> names evidence under <c>evidence/protected/</c>
/// (the capability table's rules, docs/capabilities.md).</item>
/// <item><c>experimental</c> or <c>supported</c> needs at least one SM entry with that status or better: an SM without hardware
/// evidence is <c>untested</c>. An SM entry that is not <c>unsupported</c> or <c>untested</c> names its evidence; SMs are unique.</item>
/// <item>The composition is consistent: <c>composes</c> lists no missing or pending ops, <c>composes_pending_reference</c> lists
/// pending ops only, <c>missing_ops</c> lists at least one missing op, and a recipe whose model does not compose is
/// <c>unsupported</c>.</item>
/// <item>With a repository root, every path (evidence, admission, definition project) exists and has no <c>..</c> segment.</item>
/// </list>
/// </remarks>
public sealed class RecipeValidator
{
    private const string ResourceName = "roost-recipe-v1.schema.json";

    private readonly StrictSchema _schema;
    private readonly string? _repositoryRoot;

    /// <summary>Creates a validator over the embedded schema.</summary>
    /// <param name="repositoryRoot">Repository root that recipe paths are relative to; null skips the existence checks.</param>
    public RecipeValidator(string? repositoryRoot)
    {
        _repositoryRoot = repositoryRoot;
        using var schema = JsonDocument.Parse(SchemaText());
        _schema = StrictSchema.Load(schema.RootElement.Clone());
    }

    /// <summary>The embedded schema text.</summary>
    /// <returns>The JSON Schema.</returns>
    public static string SchemaText()
    {
        using var s = typeof(RecipeValidator).Assembly.GetManifestResourceStream(ResourceName)
            ?? throw new InvalidOperationException($"Embedded resource {ResourceName} is missing.");
        using var r = new StreamReader(s);
        return r.ReadToEnd();
    }

    /// <summary>Validates files and directories (every <c>*.json</c> directly inside a directory).</summary>
    /// <param name="paths">Files or directories.</param>
    /// <returns>One "path: location: message" line per problem; empty when every recipe is valid.</returns>
    public IReadOnlyList<string> ValidatePaths(IEnumerable<string> paths)
    {
        ArgumentNullException.ThrowIfNull(paths);
        var files = new List<string>();
        var errors = new List<string>();
        foreach (string p in paths)
        {
            if (Directory.Exists(p))
            {
                files.AddRange(Directory.EnumerateFiles(p, "*.json").Order(StringComparer.Ordinal));
            }
            else if (File.Exists(p))
            {
                files.Add(p);
            }
            else
            {
                errors.Add($"{p}: : no such file or directory");
            }
        }

        var ids = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (string f in files)
        {
            string? id = null;
            foreach (string e in Validate(File.ReadAllText(f), Path.GetFileName(f), out id))
            {
                errors.Add($"{f}: {e}");
            }

            if (id is not null)
            {
                if (ids.TryGetValue(id, out string? other))
                {
                    errors.Add($"{f}: /id: '{id}' is also the id of {other}");
                }
                else
                {
                    ids.Add(id, f);
                }
            }
        }

        return errors;
    }

    /// <summary>Validates one recipe.</summary>
    /// <param name="json">The recipe text.</param>
    /// <param name="fileName">Its file name, for the name rule; null skips that rule.</param>
    /// <returns>One "location: message" line per problem; empty when valid.</returns>
    public IReadOnlyList<string> Validate(string json, string? fileName) => Validate(json, fileName, out _);

    private List<string> Validate(string json, string? fileName, out string? id)
    {
        id = null;
        var errors = new List<string>();
        JsonDocument doc;
        try
        {
            doc = JsonDocument.Parse(json);
        }
        catch (JsonException e)
        {
            errors.Add($": not valid JSON ({e.Message})");
            return errors;
        }

        using (doc)
        {
            var root = doc.RootElement;
            if (HasDuplicateKeys(root, "", errors))
            {
                return errors;
            }

            _schema.Validate(root, errors);
            if (errors.Count > 0)
            {
                // Semantic rules read the shape the schema guarantees.
                return errors;
            }

            id = root.GetProperty("id").GetString()!;
            if (fileName is not null && !string.Equals(fileName, id + ".json", StringComparison.Ordinal))
            {
                errors.Add($"/id: the file is {fileName}; a recipe with id '{id}' is named {id}.json");
            }

            string status = root.GetProperty("status").GetString()!;
            var evidence = Strings(root, "evidence");
            if (status != "unsupported" && evidence.Count == 0)
            {
                errors.Add($"/evidence: status {status} names no evidence (missing never means supported)");
            }

            if (status == "supported" && !evidence.Any(p => p.StartsWith("evidence/protected/", StringComparison.Ordinal)))
            {
                errors.Add("/evidence: status supported needs protected fixtures and tolerances under evidence/protected/");
            }

            CheckHardware(root, status, errors);
            CheckComposition(root.GetProperty("composition"), status, errors);
            CheckPaths(root, evidence, errors);
            return errors;
        }
    }

    private static void CheckHardware(JsonElement root, string status, List<string> errors)
    {
        var seen = new HashSet<int>();
        bool hardwareEvidence = false;
        if (root.TryGetProperty("hardware", out var hw))
        {
            int i = 0;
            foreach (var h in hw.EnumerateArray())
            {
                string at = "/hardware/" + i.ToString(CultureInfo.InvariantCulture);
                int sm = h.GetProperty("sm").GetInt32();
                string smStatus = h.GetProperty("status").GetString()!;
                if (!seen.Add(sm))
                {
                    errors.Add($"{at}/sm: SM {sm.ToString(CultureInfo.InvariantCulture)} is listed twice");
                }

                bool claims = smStatus is "experimental" or "supported";
                if (claims && h.GetProperty("evidence").GetArrayLength() == 0)
                {
                    errors.Add($"{at}/evidence: SM {sm.ToString(CultureInfo.InvariantCulture)} is {smStatus} without evidence");
                }

                if (Rank(smStatus) > Rank(status))
                {
                    errors.Add($"{at}/status: SM {sm.ToString(CultureInfo.InvariantCulture)} is {smStatus}, above the recipe's status {status}");
                }

                hardwareEvidence |= claims && Rank(smStatus) >= Rank(status);
                i++;
            }
        }

        if (status is "experimental" or "supported" && !hardwareEvidence)
        {
            errors.Add($"/hardware: status {status} needs an SM entry with status {status} or better; an SM without hardware evidence is untested");
        }
    }

    private static int Rank(string status) => status switch
    {
        "supported" => 3,
        "experimental" => 2,
        "untested" => 1,
        _ => 0,
    };

    private static void CheckComposition(JsonElement c, string status, List<string> errors)
    {
        string verdict = c.GetProperty("verdict").GetString()!;
        int missing = c.GetProperty("missing_ops").GetArrayLength();
        int pending = c.GetProperty("pending_ops").GetArrayLength();
        switch (verdict)
        {
            case "composes" when missing > 0 || pending > 0:
                errors.Add("/composition: verdict composes lists missing or pending ops");
                break;
            case "composes_pending_reference" when missing > 0 || pending == 0:
                errors.Add("/composition: verdict composes_pending_reference needs pending ops and no missing ops");
                break;
            case "missing_ops" when missing == 0:
                errors.Add("/composition: verdict missing_ops lists no missing ops");
                break;
            default:
                break;
        }

        if (verdict != "composes" && status != "unsupported")
        {
            errors.Add($"/status: the model does not compose (verdict {verdict}), so the recipe is unsupported, not {status}");
        }
    }

    private void CheckPaths(JsonElement root, List<string> evidence, List<string> errors)
    {
        var paths = new List<(string At, string Path)>();
        for (int i = 0; i < evidence.Count; i++)
        {
            paths.Add(("/evidence/" + i.ToString(CultureInfo.InvariantCulture), evidence[i]));
        }

        if (root.TryGetProperty("hardware", out var hw))
        {
            int i = 0;
            foreach (var h in hw.EnumerateArray())
            {
                var ev = Strings(h, "evidence");
                for (int j = 0; j < ev.Count; j++)
                {
                    paths.Add(($"/hardware/{i.ToString(CultureInfo.InvariantCulture)}/evidence/{j.ToString(CultureInfo.InvariantCulture)}", ev[j]));
                }

                i++;
            }
        }

        if (root.TryGetProperty("admission", out var adm))
        {
            paths.Add(("/admission", adm.GetString()!));
        }

        paths.Add(("/definition/project", root.GetProperty("definition").GetProperty("project").GetString()!));
        foreach (var (at, p) in paths)
        {
            if (p.Split('/').Contains(".."))
            {
                errors.Add($"{at}: '{p}' has a '..' segment");
                continue;
            }

            if (_repositoryRoot is not null)
            {
                string full = Path.Combine(_repositoryRoot, p.TrimEnd('/'));
                if (!File.Exists(full) && !Directory.Exists(full))
                {
                    errors.Add($"{at}: '{p}' does not exist in the repository");
                }
            }
        }
    }

    private static List<string> Strings(JsonElement obj, string name) =>
        obj.TryGetProperty(name, out var a) ? a.EnumerateArray().Select(e => e.GetString()!).ToList() : [];

    private static bool HasDuplicateKeys(JsonElement e, string at, List<string> errors)
    {
        bool found = false;
        if (e.ValueKind == JsonValueKind.Object)
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (var p in e.EnumerateObject())
            {
                if (!seen.Add(p.Name))
                {
                    errors.Add($"{at}: duplicate key '{p.Name}'");
                    found = true;
                }

                found |= HasDuplicateKeys(p.Value, at + "/" + p.Name, errors);
            }
        }
        else if (e.ValueKind == JsonValueKind.Array)
        {
            int i = 0;
            foreach (var x in e.EnumerateArray())
            {
                found |= HasDuplicateKeys(x, at + "/" + i.ToString(CultureInfo.InvariantCulture), errors);
                i++;
            }
        }

        return found;
    }
}
