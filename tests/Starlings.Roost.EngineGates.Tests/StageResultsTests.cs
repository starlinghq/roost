using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Roost.EngineGates.Tests.Support;

namespace Starlings.Roost.EngineGates.Tests;

/// <summary>
/// The gates as stage results, format v3: both documents of each model are built from a full gate run, validated against
/// evidence/schemas/stage-results.schema.json, and written to a run directory the evaluator reads
/// (<c>roost-evidence evaluate</c>). The directory is <c>$ROOST_ENGINE_GATES_OUT/&lt;model&gt;</c> when that variable is set,
/// otherwise <c>engine-gates/&lt;model&gt;</c> beside the test assembly.
/// </summary>
public sealed class StageResultsTests
{
    private static readonly Lazy<SchemaValidator> s_validator = new(() =>
        new SchemaValidator(JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "stage-results.schema.json"))).RootElement));

    public static TheoryData<TinyProfile> Profiles => [TinyProfile.Dense, TinyProfile.Hybrid];

    [Theory]
    [MemberData(nameof(Profiles))]
    public async Task GateRowsAreValidFormatV3AndEveryGatePasses(TinyProfile profile)
    {
        var run = await GateRun.CollectAsync(profile, TestContext.Current.CancellationToken);
        var template = StageResults.Template(profile);
        var dotnet = StageResults.Dotnet(run);

        AssertValid(template);
        AssertValid(dotnet);

        var sections = dotnet["sections"]!;
        Assert.True((bool)sections["cached_vs_uncached"]!["tokens_equal"]!);
        Assert.All(sections["chunked_prefill"]!.AsObject(), p => Assert.True((bool)p.Value!["tokens_equal_whole"]!, p.Key));
        Assert.All(sections["concurrency"]!["results"]!.AsObject(), p => Assert.True((bool)p.Value!, p.Key));
        foreach (var (id, row) in dotnet["cases"]!.AsObject())
        {
            Assert.True((bool)row!["stream_text_equals_result"]! && (bool)row["stream_ids_equal_result"]! && (bool)row["stream_text_equals_decode"]!, id);
        }

        string root = Environment.GetEnvironmentVariable("ROOST_ENGINE_GATES_OUT") is { Length: > 0 } dir ? dir : Path.Combine(AppContext.BaseDirectory, "engine-gates");
        string target = Path.Combine(root, Family(profile));
        StageResults.Write(target, run);
        TestContext.Current.SendDiagnosticMessage($"WROTE {target}");
        AssertValid(JsonNode.Parse(File.ReadAllText(Path.Combine(target, "dotnet.results.json")))!.AsObject());
    }

    [Fact]
    public void TheValidatorRejectsWhatTheSchemaRejects()
    {
        var good = SampleDocument("dotnet");
        AssertValid(good);

        var wrongVersion = Clone(good);
        wrongVersion["format_version"] = 2;
        var noSections = Clone(good);
        noSections.Remove("sections");
        var extra = Clone(good);
        extra["sections"]!["preemption"] = new JsonObject();
        var badTop = Clone(good);
        badTop["cases"]!["short"]!["step0_top"] = new JsonArray(new JsonArray(1));
        var badFinish = Clone(good);
        badFinish["cases"]!["short"]!["finish_reason"] = "stop";
        var badProfile = Clone(good);
        badProfile["profile"] = "Roost Tiny";

        foreach (var broken in new[] { wrongVersion, noSections, extra, badTop, badFinish, badProfile })
        {
            Assert.NotEmpty(Validate(broken));
        }
    }

    /// <summary>The sample run committed for the evaluator's pytest has the shape this harness writes today.</summary>
    /// <param name="profile">The model.</param>
    [Theory]
    [MemberData(nameof(Profiles))]
    public void TheCommittedSampleRunMatchesTheHarness(TinyProfile profile)
    {
        foreach (var stage in new[] { "template", "dotnet" })
        {
            var sample = SampleDocument(stage, profile);
            AssertValid(sample);
            Assert.Equal(StageResults.ProfileId(profile), (string?)sample["profile"]);
            var cases = sample["cases"]!.AsObject();
            Assert.Equal(GateCases.All.Select(c => c.Id).Order(StringComparer.Ordinal), cases.Select(p => p.Key).Order(StringComparer.Ordinal));
            foreach (var c in GateCases.All)
            {
                Assert.Equal(c.PromptIds, cases[c.Id]!["prompt_ids"]!.AsArray().Select(n => (int)n!));
            }

            if (stage == "dotnet")
            {
                var sections = sample["sections"]!;
                Assert.Equal(GateCases.Cached.Id, (string?)sections["cached_vs_uncached"]!["case"]);
                var keys = GateCases.All.SelectMany(c => Gates.ChunkSizes.Where(k => k < c.PromptIds.Length).Select(k => $"{c.Id}/{k}"));
                Assert.Equal(keys.Order(StringComparer.Ordinal), sections["chunked_prefill"]!.AsObject().Select(p => p.Key).Order(StringComparer.Ordinal));
            }
        }
    }

    private static string Family(TinyProfile profile) => StageResults.ProfileId(profile).Split('/')[0];

    private static JsonObject SampleDocument(string stage, TinyProfile profile = TinyProfile.Dense) =>
        JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "engine-gates", Family(profile), $"{stage}.results.json")))!.AsObject();

    private static JsonObject Clone(JsonObject o) => JsonNode.Parse(o.ToJsonString())!.AsObject();

    private static List<string> Validate(JsonObject document) =>
        s_validator.Value.Validate(JsonDocument.Parse(document.ToJsonString()).RootElement);

    private static void AssertValid(JsonObject document)
    {
        var errors = Validate(document);
        Assert.True(errors.Count == 0, string.Join("\n", errors));
    }
}
