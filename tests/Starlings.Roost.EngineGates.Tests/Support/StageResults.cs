using System.Text.Json;
using System.Text.Json.Nodes;

namespace Starlings.Roost.EngineGates.Tests.Support;

/// <summary>Every gate's measurements for one model, gathered for the stage-results documents.</summary>
/// <param name="Profile">The model.</param>
/// <param name="Baseline">Baseline output per case.</param>
/// <param name="Chunked">Chunked prefill per <c>&lt;case&gt;/&lt;chunk&gt;</c>.</param>
/// <param name="Cached">Cached against a fresh prefill.</param>
/// <param name="Batched">Every case batched (one plan in flight).</param>
/// <param name="Pipelined">Every case batched (two plans in flight).</param>
/// <param name="Preempted">Every case on a pool that forces preemption.</param>
/// <param name="Streaming">Streaming against non-streaming per case.</param>
internal sealed record GateRun(
    TinyProfile Profile,
    IReadOnlyDictionary<string, CaseOutput> Baseline,
    IReadOnlyDictionary<string, Comparison> Chunked,
    CachedVsUncached Cached,
    TogetherRun Batched,
    TogetherRun Pipelined,
    TogetherRun Preempted,
    IReadOnlyDictionary<string, StreamingResult> Streaming)
{
    /// <summary>Runs every gate for <paramref name="profile"/>.</summary>
    /// <param name="profile">The model.</param>
    /// <param name="cancellationToken">Cancels the streaming runs.</param>
    /// <returns>The measurements.</returns>
    public static async Task<GateRun> CollectAsync(TinyProfile profile, CancellationToken cancellationToken)
    {
        var chunked = new SortedDictionary<string, Comparison>(StringComparer.Ordinal);
        foreach (var c in GateCases.All)
        {
            foreach (int k in Gates.ChunkSizes.Where(k => k < c.PromptIds.Length))
            {
                chunked[$"{c.Id}/{k}"] = Gates.Chunked(profile, c, k).Result;
            }
        }

        return new GateRun(
            profile,
            Gates.Baseline(profile),
            chunked,
            Gates.CachedAgainstFreshPrefill(profile),
            Gates.Concurrent(profile, 1),
            Gates.Concurrent(profile, 2),
            Gates.Preempted(profile),
            await Gates.Streaming(profile, cancellationToken));
    }
}

/// <summary>
/// Writes the gate measurements as stage results, format v3, task generation (evidence/schemas/stage-results.schema.json):
/// <c>template.results.json</c> (each case's prompt as rendered, raw completion with no chat template, and its ids) and
/// <c>dotnet.results.json</c> (the baseline generation of each case with its step-0 top 5, the streaming flags and usage,
/// and the <c>cached_vs_uncached</c>, <c>chunked_prefill</c> and <c>concurrency</c> sections).
/// </summary>
/// <remarks>
/// Format v3 has no section for preemption or for recurrent snapshot restore, and its contract owner is stream B2. Until it
/// has one, the gates fold them into the sections that exist: a case's <c>concurrency</c> flag is true only when its tokens
/// equal the single, uninterrupted run in the batched run, the pipelined batched run and the preempted run; the hybrid
/// model's <c>cached_vs_uncached</c> case starts from a restored snapshot. The separate checks are the tests in
/// <see cref="EngineGateTests"/>.
/// </remarks>
internal static class StageResults
{
    /// <summary>The producer name recorded in every document.</summary>
    public const string Producer = "Starlings.Roost.EngineGates.Tests";

    /// <summary>The candidate name recorded in every document.</summary>
    public const string Candidate = "roost-engine-cpuops";

    /// <summary>Indented, as the evaluator writes its own files.</summary>
    public static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    /// <summary>The <c>&lt;family&gt;/&lt;profile&gt;</c> id of a tiny model's CPU run.</summary>
    /// <param name="profile">The model.</param>
    /// <returns>The id.</returns>
    public static string ProfileId(TinyProfile profile) => profile == TinyProfile.Dense ? "roost-tiny/cpuops-f32" : "roost-tiny-hybrid/cpuops-bf16";

    /// <summary>The template stage document.</summary>
    /// <param name="profile">The model.</param>
    /// <returns>The document.</returns>
    public static JsonObject Template(TinyProfile profile)
    {
        var cases = new JsonObject();
        foreach (var c in GateCases.All)
        {
            cases[c.Id] = new JsonObject
            {
                ["rendered"] = c.Prompt,
                ["render_error"] = null,
                ["prompt_ids"] = Ids(c.PromptIds),
            };
        }

        var doc = Head(profile, "template");
        doc["cases"] = cases;
        return doc;
    }

    /// <summary>The dotnet stage document.</summary>
    /// <param name="run">The measurements.</param>
    /// <returns>The document.</returns>
    public static JsonObject Dotnet(GateRun run)
    {
        var cases = new JsonObject();
        foreach (var c in GateCases.All)
        {
            var output = run.Baseline[c.Id];
            var stream = run.Streaming[c.Id];
            var logprobs = Comparison.LogSoftmax(output.Logits[0]);
            var top = new JsonArray();
            foreach (int t in Enumerable.Range(0, logprobs.Length).OrderBy(t => -logprobs[t]).ThenBy(t => t).Take(5))
            {
                top.Add(new JsonArray(t, logprobs[t]));
            }

            cases[c.Id] = new JsonObject
            {
                ["prompt_ids"] = Ids(c.PromptIds),
                ["output_ids"] = Ids(output.Tokens),
                ["finish_reason"] = "length",
                ["step0_top"] = top,
                ["stream_text_equals_result"] = stream.StreamTextEqualsResult,
                ["stream_ids_equal_result"] = stream.StreamIdsEqualResult,
                ["stream_text_equals_decode"] = stream.StreamTextEqualsDecode,
                ["usage"] = new JsonObject { ["prompt_tokens"] = stream.PromptTokens, ["completion_tokens"] = stream.CompletionTokens },
            };
        }

        var chunked = new JsonObject();
        foreach (var (key, r) in run.Chunked)
        {
            chunked[key] = new JsonObject { ["step0_max_abs_logprob_vs_whole"] = r.Step0MaxAbsLogprob, ["tokens_equal_whole"] = r.TokensEqual };
        }

        var concurrency = new JsonObject();
        foreach (var c in GateCases.All)
        {
            concurrency[c.Id] = run.Batched.Results[c.Id].TokensEqual && run.Pipelined.Results[c.Id].TokensEqual && run.Preempted.Results[c.Id].TokensEqual;
        }

        var doc = Head(run.Profile, "dotnet");
        doc["cases"] = cases;
        doc["sections"] = new JsonObject
        {
            ["cached_vs_uncached"] = new JsonObject
            {
                ["case"] = run.Cached.Case,
                ["steps"] = run.Cached.Steps,
                ["max_abs_logprob"] = run.Cached.MaxAbsLogprob,
                ["tokens_equal"] = run.Cached.TokensEqual,
            },
            ["chunked_prefill"] = chunked,
            ["concurrency"] = new JsonObject { ["results"] = concurrency },
        };
        return doc;
    }

    /// <summary>Writes both documents under <paramref name="directory"/>, as the evaluator's run directory expects them.</summary>
    /// <param name="directory">The run directory (created).</param>
    /// <param name="run">The measurements.</param>
    public static void Write(string directory, GateRun run)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "template.results.json"), Template(run.Profile).ToJsonString(Indented) + "\n");
        File.WriteAllText(Path.Combine(directory, "dotnet.results.json"), Dotnet(run).ToJsonString(Indented) + "\n");
    }

    private static JsonObject Head(TinyProfile profile, string stage) => new()
    {
        ["format"] = "roost.stage-results",
        ["format_version"] = 3,
        ["task"] = "generation",
        ["profile"] = ProfileId(profile),
        ["stage"] = stage,
        ["candidate"] = Candidate,
        ["producer"] = new JsonObject { ["name"] = Producer, ["repository_commit"] = null },
        ["environment"] = new JsonObject { ["provider"] = "cpu", ["provider_options"] = new JsonObject() },
    };

    private static JsonArray Ids(IEnumerable<int> ids) => [.. ids.Select(i => (JsonNode)i)];
}
