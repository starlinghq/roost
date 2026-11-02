using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Roost.Onboarding;

namespace Starlings.Roost.Onboarding.Tests;

public sealed class RecipeValidatorTests
{
    internal static string RepositoryRoot()
    {
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
        {
            if (File.Exists(Path.Combine(d.FullName, "Roost.slnx")))
            {
                return d.FullName;
            }
        }

        throw new InvalidOperationException("Roost.slnx not found above the test output directory.");
    }

    private static string RecipesDirectory() => Path.Combine(RepositoryRoot(), "evidence", "recipes");

    private static JsonNode ValidRecipe() => JsonNode.Parse(File.ReadAllText(Path.Combine(RecipesDirectory(), "qwen3.5-0.8b.json")))!;

    private static IReadOnlyList<string> Validate(JsonNode recipe, string? fileName = "qwen3.5-0.8b.json") =>
        new RecipeValidator(RepositoryRoot()).Validate(recipe.ToJsonString(), fileName);

    [Fact]
    public void EveryCommittedRecipeIsValid()
    {
        Assert.NotEmpty(Directory.EnumerateFiles(RecipesDirectory(), "*.json"));
        Assert.Empty(new RecipeValidator(RepositoryRoot()).ValidatePaths([RecipesDirectory()]));
    }

    [Fact]
    public void TheQwen35RecipeCompositionIsWhatInspectReportsForItsConfig()
    {
        var recipe = ValidRecipe();
        var r = ConfigInspector.InspectFile(ConfigInspectorTests.DataPath("qwen3.5-0.8b.config.json"));
        var c = recipe["composition"]!;

        Assert.Equal(c["verdict"]!.GetValue<string>(), JsonSerializer.Serialize(r.Verdict, OnboardingJsonContext.Default.CompositionVerdict).Trim('"'));
        Assert.Equal(c["contract_version"]!.GetValue<int>(), r.ContractVersion);
        Assert.Equal(c["missing_ops"]!.AsArray().Select(n => n!.GetValue<string>()), r.MissingOps);
        Assert.Equal(c["pending_ops"]!.AsArray().Select(n => n!.GetValue<string>()), r.PendingOps);
        Assert.Equal(recipe["model"]!["model_type"]!.GetValue<string>(), r.ModelType);
    }

    [Fact]
    public void TheSchemaIsEmbeddedAndStrictEverywhere()
    {
        using var doc = JsonDocument.Parse(RecipeValidator.SchemaText());
        int objects = 0;
        void Walk(JsonElement e)
        {
            if (e.ValueKind == JsonValueKind.Object)
            {
                if (e.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String && t.GetString() == "object")
                {
                    objects++;
                    Assert.True(e.TryGetProperty("additionalProperties", out var ap) && ap.ValueKind == JsonValueKind.False, "every object schema sets additionalProperties false");
                }

                foreach (var p in e.EnumerateObject())
                {
                    Walk(p.Value);
                }
            }
            else if (e.ValueKind == JsonValueKind.Array)
            {
                foreach (var x in e.EnumerateArray())
                {
                    Walk(x);
                }
            }
        }

        Walk(doc.RootElement);
        Assert.Equal(6, objects);
    }

    [Fact]
    public void ASchemaKeywordOutsideTheSubsetIsRefused()
    {
        using var doc = JsonDocument.Parse("""{ "type": "object", "properties": { "a": { "oneOf": [ { "type": "string" } ] } } }""");
        var e = Assert.Throws<InvalidOperationException>(() => StrictSchema.Load(doc.RootElement));
        Assert.Contains("oneOf", e.Message, StringComparison.Ordinal);

        using var badRef = JsonDocument.Parse("""{ "$ref": "#/$defs/nothing" }""");
        Assert.Throws<InvalidOperationException>(() => StrictSchema.Load(badRef.RootElement));
    }

    public static TheoryData<string, string, string> SchemaViolations() => new()
    {
        { "unknown top-level property", """{"op":"add","path":"extra","value":1}""", ": unknown property 'extra'" },
        { "unknown nested property", """{"op":"add","path":"model/branch","value":"main"}""", "/model: unknown property 'branch'" },
        { "branch instead of SHA", """{"op":"set","path":"model/revision","value":"main"}""", "/model/revision: \"main\" does not match" },
        { "short SHA", """{"op":"set","path":"model/revision","value":"2fc0636"}""", "/model/revision:" },
        { "missing status", """{"op":"remove","path":"status"}""", ": 'status' is required" },
        { "unknown status", """{"op":"set","path":"status","value":"beta"}""", "/status: \"beta\" is not one of" },
        { "wrong plugin", """{"op":"set","path":"plugin","value":"other"}""", "/plugin: must be \"roost-v1\"" },
        { "schema version", """{"op":"set","path":"schema_version","value":2}""", "/schema_version: must be 1" },
        { "block size zero", """{"op":"set","path":"serving/block_size","value":0}""", "/serving/block_size: 0 is less than 1" },
        { "fractional block size", """{"op":"set","path":"serving/block_size","value":16.5}""", "/serving/block_size: 16.5 is not of type integer" },
        { "unknown SM", """{"op":"set","path":"hardware/0/sm","value":75}""", "/hardware/0/sm: 75 is not one of" },
        { "duplicate evidence", """{"op":"set","path":"evidence","value":["docs/m0/roost-r022-cpu-parity.md","docs/m0/roost-r022-cpu-parity.md"]}""", "/evidence/1: duplicates item 0" },
        { "absolute path", """{"op":"set","path":"evidence/0","value":"/etc/passwd"}""", "/evidence/0: \"/etc/passwd\" does not match" },
        { "definition outside Models", """{"op":"set","path":"definition/project","value":"src/Starlings.Roost.Engine"}""", "/definition/project:" },
        { "bad op name", """{"op":"set","path":"composition/pending_ops","value":["Mla Attention"]}""", "/composition/pending_ops/0:" },
    };

    [Theory]
    [MemberData(nameof(SchemaViolations))]
    public void SchemaViolationsAreReportedWithTheirLocation(string name, string patch, string expected)
    {
        var errors = Validate(Apply(ValidRecipe(), patch));
        Assert.True(errors.Any(e => e.StartsWith(expected, StringComparison.Ordinal) || e.Contains(expected, StringComparison.Ordinal)), $"{name}: {string.Join(" | ", errors)}");
    }

    public static TheoryData<string, string, string> RuleViolations() => new()
    {
        { "experimental without hardware", """{"op":"remove","path":"hardware"}""", "/hardware: status experimental needs an SM entry" },
        { "experimental with untested hardware only", """{"op":"set","path":"hardware/0/status","value":"untested"}""", "/hardware: status experimental needs" },
        { "SM above the recipe", """{"op":"set","path":"status","value":"untested"}""", "/hardware/0/status: SM 86 is experimental, above the recipe's status untested" },
        { "supported without protected evidence", """{"op":"set","path":"status","value":"supported"}""", "/evidence: status supported needs protected fixtures" },
        { "status without evidence", """{"op":"set","path":"evidence","value":[]}""", "/evidence: status experimental names no evidence" },
        { "SM claim without evidence", """{"op":"set","path":"hardware/0/evidence","value":[]}""", "/hardware/0/evidence: SM 86 is experimental without evidence" },
        { "duplicate SM", """{"op":"set","path":"hardware/1/sm","value":86}""", "/hardware/1/sm: SM 86 is listed twice" },
        { "composes with missing ops", """{"op":"set","path":"composition/missing_ops","value":["MlaAttention"]}""", "/composition: verdict composes lists missing or pending ops" },
        { "missing_ops verdict without ops", """{"op":"set","path":"composition/verdict","value":"missing_ops"}""", "/composition: verdict missing_ops lists no missing ops" },
        { "pending verdict without ops", """{"op":"set","path":"composition/verdict","value":"composes_pending_reference"}""", "/composition: verdict composes_pending_reference needs pending ops" },
        { "non-composing model not unsupported", """{"op":"set","path":"composition/verdict","value":"missing_ops"}""", "/status: the model does not compose" },
        { "evidence that does not exist", """{"op":"set","path":"evidence/0","value":"docs/m0/no-such-report.md"}""", "/evidence/0: 'docs/m0/no-such-report.md' does not exist" },
        { "dot-dot segment", """{"op":"set","path":"evidence/0","value":"docs/../../etc"}""", "/evidence/0: 'docs/../../etc' has a '..' segment" },
        { "definition that does not exist", """{"op":"set","path":"definition/project","value":"src/Starlings.Roost.Models.Nothing"}""", "/definition/project: 'src/Starlings.Roost.Models.Nothing' does not exist" },
        { "admission that does not exist", """{"op":"set","path":"admission","value":"evidence/registry/none.toml"}""", "/admission: 'evidence/registry/none.toml' does not exist" },
        { "file not named after id", """{"op":"set","path":"id","value":"qwen3.5-0.8b-other"}""", "/id: the file is qwen3.5-0.8b.json" },
    };

    [Theory]
    [MemberData(nameof(RuleViolations))]
    public void RulesBeyondTheSchemaAreEnforced(string name, string patch, string expected)
    {
        var errors = Validate(Apply(ValidRecipe(), patch));
        Assert.True(errors.Any(e => e.StartsWith(expected, StringComparison.Ordinal)), $"{name}: {string.Join(" | ", errors)}");
    }

    [Fact]
    public void AnUnsupportedRecipeNeedsNoEvidence()
    {
        var recipe = ValidRecipe();
        recipe["status"] = "unsupported";
        recipe["evidence"] = new JsonArray();
        recipe.AsObject().Remove("hardware");
        recipe["composition"]!["verdict"] = "missing_ops";
        recipe["composition"]!["missing_ops"] = new JsonArray("MoeRouteTopK", "Mixer(mamba)");
        Assert.Empty(Validate(recipe));
    }

    [Fact]
    public void DuplicateKeysAndInvalidJsonAreErrors()
    {
        var v = new RecipeValidator(null);
        Assert.Contains(v.Validate("""{"id":"a","id":"b"}""", null), e => e.Contains("duplicate key 'id'", StringComparison.Ordinal));
        Assert.Contains(v.Validate("{", null), e => e.Contains("not valid JSON", StringComparison.Ordinal));
    }

    [Fact]
    public void IdsAreUniqueAcrossFilesValidatedTogether()
    {
        string dir = Directory.CreateTempSubdirectory("roost-recipes-").FullName;
        try
        {
            string a = Path.Combine(dir, "a");
            string b = Path.Combine(dir, "b");
            Directory.CreateDirectory(a);
            Directory.CreateDirectory(b);
            string text = ValidRecipe().ToJsonString();
            File.WriteAllText(Path.Combine(a, "qwen3.5-0.8b.json"), text);
            File.WriteAllText(Path.Combine(b, "qwen3.5-0.8b.json"), text);

            var errors = new RecipeValidator(RepositoryRoot()).ValidatePaths([a, b, Path.Combine(dir, "missing")]);

            Assert.Contains(errors, e => e.Contains("/id: 'qwen3.5-0.8b' is also the id of", StringComparison.Ordinal));
            Assert.Contains(errors, e => e.Contains("no such file or directory", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    // A tiny patch format for the theory data: set, add or remove one value at a slash-separated path.
    private static JsonNode Apply(JsonNode recipe, string patch)
    {
        var p = JsonNode.Parse(patch)!;
        string op = p["op"]!.GetValue<string>();
        string[] path = p["path"]!.GetValue<string>().Split('/');
        JsonNode parent = recipe;
        foreach (string seg in path[..^1])
        {
            parent = int.TryParse(seg, out int i) ? parent[i]! : parent[seg]!;
        }

        string last = path[^1];
        JsonNode? value = p["value"]?.DeepClone();
        if (op == "remove")
        {
            parent.AsObject().Remove(last);
        }
        else if (int.TryParse(last, out int index))
        {
            parent.AsArray()[index] = value;
        }
        else
        {
            parent[last] = value;
        }

        return recipe;
    }
}
