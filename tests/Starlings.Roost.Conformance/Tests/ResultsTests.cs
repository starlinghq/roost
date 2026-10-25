using System.Text.Json;
using System.Text.RegularExpressions;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance.Tests;

public sealed class ResultsTests
{
    // The evaluator rejects these keys in results (evidence/pipeline stage_results.FORBIDDEN_KEYS).
    private static readonly string[] Forbidden = ["tolerance", "tolerances", "threshold", "thresholds"];

    private static JsonDocument Render(IKernelBackend b, RunOptions o, string stage = "cpu")
    {
        ConformanceReport r = ConformanceRunner.Run(b, o);
        var env = new RunEnvironment { Stage = stage, Subject = "oracle", Candidate = "Oracle against CpuOps", RepositoryCommit = new string('a', 40) };
        env.Assets.Add(("op-class-tolerances.proposal.json", o.Tolerances.Sha256));
        env.Options["sm"] = "0";
        return JsonDocument.Parse(ResultsWriter.Render(r, env));
    }

    private static IEnumerable<string> Keys(JsonElement e)
    {
        if (e.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty p in e.EnumerateObject())
            {
                yield return p.Name;
                foreach (string k in Keys(p.Value))
                {
                    yield return k;
                }
            }
        }
        else if (e.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement x in e.EnumerateArray())
            {
                foreach (string k in Keys(x))
                {
                    yield return k;
                }
            }
        }
    }

    [Fact]
    public void TheDocumentHasTheStageResultsEnvelope()
    {
        using var oracle = new OracleBackend();
        using JsonDocument d = Render(oracle, Fixtures.Options(ops: new HashSet<OpId> { OpId.Residual, OpId.Embedding }));
        JsonElement r = d.RootElement;
        Assert.Equal("roost.stage-results", r.GetProperty("format").GetString());
        Assert.Equal(3, r.GetProperty("format_version").GetInt32());
        Assert.Equal("kernel-conformance", r.GetProperty("task").GetString());
        Assert.Equal("cpu", r.GetProperty("stage").GetString());
        Assert.Matches("^kernels/[a-z0-9][a-z0-9.-]*$", r.GetProperty("profile").GetString()!);
        Assert.Matches("^[0-9a-f]{40}$", r.GetProperty("producer").GetProperty("repository_commit").GetString()!);
        Assert.All(r.GetProperty("assets").EnumerateArray(), a => Assert.Matches("^[0-9a-f]{64}$", a.GetProperty("sha256").GetString()!));
        Assert.True(r.GetProperty("sections").GetProperty("summary").GetProperty("green").GetBoolean());
    }

    [Fact]
    public void ResultsCarryMeasurementsAndNoToleranceKeys()
    {
        using var oracle = new OracleBackend();
        using JsonDocument d = Render(oracle, Fixtures.Options(ops: new HashSet<OpId> { OpId.RmsNorm }));
        Assert.DoesNotContain(Keys(d.RootElement), k => Forbidden.Contains(k, StringComparer.OrdinalIgnoreCase));
        JsonProperty first = d.RootElement.GetProperty("cases").EnumerateObject().First(p => p.Value.GetProperty("outcome").GetString() == "pass");
        foreach (string name in new[] { "max_abs", "max_utilisation", "violations", "nonfinite", "ref_rms", "count" })
        {
            Assert.True(first.Value.GetProperty("outputs")[0].TryGetProperty(name, out _), name);
        }

        Assert.Equal(first.Value.GetProperty("class").GetString(), d.RootElement.GetProperty("sections").GetProperty("classes").EnumerateObject().First(c => c.Name == first.Value.GetProperty("class").GetString()).Name);
    }

    [Fact]
    public void CaseKeysAreUniqueAndNameTheOp()
    {
        using var oracle = new OracleBackend();
        ConformanceReport r = ConformanceRunner.Run(oracle, Fixtures.Options(ops: new HashSet<OpId> { OpId.RmsNorm, OpId.SiluAndMul, OpId.Residual, OpId.Embedding, OpId.CacheWrite }, determinism: false, invariance: false));
        var env = new RunEnvironment { Stage = "cpu", Subject = "oracle", Candidate = "x" };
        // Utf8JsonWriter keeps duplicate property names; the document would be ambiguous, so count them from the report.
        string[] keys = [.. r.Results.Select(x => x.Op + "/" + x.Variant + "/" + x.Case)];
        Assert.Equal(keys.Length, keys.Distinct().Count());
        using JsonDocument d = JsonDocument.Parse(ResultsWriter.Render(r, env));
        Assert.Equal(keys.Length, d.RootElement.GetProperty("cases").EnumerateObject().Count());
    }

    [Fact]
    public void AnInfiniteDifferenceIsWrittenAsAFiniteNumber()
    {
        using var cpu = new CpuBackend { Perturb = (c, r) => r.Outputs["out"].Raw()[0] ^= 0xff };
        using JsonDocument d = Render(cpu, Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: new HashSet<OpId> { OpId.Embedding }, determinism: false, invariance: false));
        Assert.Contains(d.RootElement.GetProperty("cases").EnumerateObject(), p => p.Value.GetProperty("outcome").GetString() == "fail");
        Assert.False(d.RootElement.GetProperty("sections").GetProperty("summary").GetProperty("green").GetBoolean());
    }

    [Fact]
    public void NotRunCasesAreRecordedAndNeverCountAsPasses()
    {
        using var cpu = new CpuBackend();
        using JsonDocument d = Render(cpu, Fixtures.Options(variants: new HashSet<string> { CpuBackend.VariantId }, ops: new HashSet<OpId> { OpId.Residual }));
        JsonElement s = d.RootElement.GetProperty("sections").GetProperty("summary");
        Assert.True(s.GetProperty("not_run").GetInt32() >= 1);
        Assert.False(s.GetProperty("goldens_given").GetBoolean());
        Assert.Equal(s.GetProperty("passed").GetInt32(), d.RootElement.GetProperty("cases").EnumerateObject().Count(p => p.Value.GetProperty("outcome").GetString() == "pass"));
    }

    [Fact]
    public void TheSchemaFileShipsAndNamesTheTask()
    {
        using JsonDocument s = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "conformance-results.schema.json")));
        Assert.Equal("kernel-conformance", s.RootElement.GetProperty("properties").GetProperty("task").GetProperty("const").GetString());
        Assert.Matches(new Regex("^https://"), s.RootElement.GetProperty("$id").GetString()!);
    }
}
