using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>Facts about the run that are not in the report.</summary>
public sealed class RunEnvironment
{
    /// <summary>"cpu" or "cuda".</summary>
    public required string Stage { get; init; }

    /// <summary>Profile slug after "kernels/": the library or reference name.</summary>
    public required string Subject { get; init; }

    /// <summary>What was tested: a library path with its digest, or "CpuOps".</summary>
    public required string Candidate { get; init; }

    /// <summary>Commit of the repository the suite was built from, 40 hex digits, or null.</summary>
    public string? RepositoryCommit { get; init; }

    /// <summary>Files the run loaded, with their digests.</summary>
    public List<(string Path, string Sha256)> Assets { get; } = [];

    /// <summary>Free-form facts (device, SM, driver, toolkit) recorded under <c>environment.provider_options</c>.</summary>
    public Dictionary<string, string> Options { get; } = new(StringComparer.Ordinal);

    /// <summary>Digest of a file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>Lower-case hex SHA-256.</returns>
    public static string Digest(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(fs));
    }
}

/// <summary>
/// Writes a report in the evaluator's stage-results envelope (format <c>roost.stage-results</c>, version 3: format, task, profile,
/// stage, candidate, producer, environment, assets, cases, sections). The task is <c>kernel-conformance</c> and the stages are
/// <c>cpu</c> and <c>cuda</c>: the committed <c>evidence/schemas/stage-results.schema.json</c> allows task <c>generation</c> only,
/// so <c>conformance-results.schema.json</c> next to the suite describes this document and a schema change is requested from
/// stream B2. Results hold measurements only (no keys the evaluator rejects as tolerances).
/// </summary>
public static class ResultsWriter
{
    /// <summary>The task value.</summary>
    public const string Task = "kernel-conformance";

    /// <summary>Writes the document.</summary>
    /// <param name="report">The report.</param>
    /// <param name="env">The run's environment.</param>
    /// <param name="path">Output file.</param>
    public static void Write(ConformanceReport report, RunEnvironment env, string path)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            Write(w, report, env);
        }

        File.WriteAllBytes(path, ms.ToArray());
    }

    /// <summary>Renders the document as text.</summary>
    /// <param name="report">The report.</param>
    /// <param name="env">The run's environment.</param>
    /// <returns>The JSON text.</returns>
    public static string Render(ConformanceReport report, RunEnvironment env)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            Write(w, report, env);
        }

        return Encoding.UTF8.GetString(ms.ToArray());
    }

    private static void Number(Utf8JsonWriter w, string name, double d)
    {
        // JSON has no infinity: a comparison that is infinitely off the limit is written as the largest double.
        w.WriteNumber(name, double.IsFinite(d) ? d : (d > 0 ? double.MaxValue : double.MinValue));
    }

    private static void Write(Utf8JsonWriter w, ConformanceReport report, RunEnvironment env)
    {
        RunOptions o = report.Options;
        w.WriteStartObject();
        w.WriteString("format", "roost.stage-results");
        w.WriteNumber("format_version", 3);
        w.WriteString("task", Task);
        w.WriteString("profile", "kernels/" + env.Subject);
        w.WriteString("stage", env.Stage);
        w.WriteString("candidate", env.Candidate);
        w.WriteStartObject("producer");
        w.WriteString("name", "Starlings.Roost.Conformance");
        if (env.RepositoryCommit is null)
        {
            w.WriteNull("repository_commit");
        }
        else
        {
            w.WriteString("repository_commit", env.RepositoryCommit);
        }

        w.WriteEndObject();
        w.WriteStartObject("environment");
        w.WriteString("machine", Environment.MachineName);
        w.WriteString("provider", env.Stage);
        w.WriteStartObject("provider_options");
        foreach ((string k, string v) in env.Options.OrderBy(k => k.Key, StringComparer.Ordinal))
        {
            w.WriteString(k, v);
        }

        w.WriteEndObject();
        w.WriteEndObject();

        w.WriteStartArray("assets");
        foreach ((string path, string sha) in env.Assets)
        {
            w.WriteStartObject();
            w.WriteString("path", path);
            w.WriteString("sha256", sha);
            w.WriteEndObject();
        }

        w.WriteEndArray();

        w.WriteStartObject("cases");
        foreach (CaseResult r in report.Results)
        {
            w.WriteStartObject(r.Op + "/" + r.Variant + "/" + r.Case);
            w.WriteString("op", r.Op.ToString());
            w.WriteString("variant", r.Variant);
            w.WriteString("dtype", r.Dtype.ToString());
            w.WriteString("source", r.Source);
            w.WriteString("class", r.Class);
            w.WriteString("outcome", r.Outcome);
            w.WriteString("reference", r.Reference);
            w.WriteString("seed", r.Seed.ToString(CultureInfo.InvariantCulture));
            w.WriteStartArray("outputs");
            foreach (OutputResult x in r.Outputs)
            {
                w.WriteStartObject();
                w.WriteString("name", x.Name);
                w.WriteNumber("count", x.Stat.Count);
                Number(w, "max_abs", x.Stat.MaxAbs);
                Number(w, "max_utilisation", x.Stat.MaxUtilisation);
                w.WriteNumber("violations", x.Stat.Violations);
                w.WriteNumber("nonfinite", x.Stat.NonFinite);
                Number(w, "ref_rms", x.Stat.RefRms);
                Number(w, "scale", x.Stat.Scale);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteString("determinism", r.Determinism);
            w.WriteString("batch_invariance", r.Invariance);
            w.WriteStartArray("checks");
            foreach (BackendCheck ch in r.Checks)
            {
                w.WriteStartObject();
                w.WriteString("name", ch.Name);
                w.WriteBoolean("ok", ch.Ok);
                w.WriteString("detail", ch.Detail);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteString("detail", r.Detail);
            w.WriteEndObject();
        }

        w.WriteEndObject();

        w.WriteStartObject("sections");
        w.WriteStartObject("summary");
        w.WriteString("backend", report.Backend);
        w.WriteString("case_profile", o.Profile.Name);
        w.WriteString("base_seed", o.BaseSeed.ToString(CultureInfo.InvariantCulture));
        w.WriteNumber("random_cases_per_op_and_dtype", o.Profile.RandomPerOp);
        w.WriteNumber("passed", report.Count("pass"));
        w.WriteNumber("failed", report.Count("fail"));
        w.WriteNumber("errors", report.Count("error"));
        w.WriteNumber("not_run", report.Count("not_run"));
        w.WriteBoolean("green", report.Green);
        w.WriteBoolean("determinism_checked", o.Determinism);
        w.WriteBoolean("batch_invariance_checked", o.Invariance);
        w.WriteBoolean("goldens_given", o.Goldens is not null);
        if (o.Goldens is not null)
        {
            w.WriteString("goldens_sha256", o.Goldens.Sha256);
        }

        w.WriteString("tolerance_file_sha256", o.Tolerances.Sha256);
        w.WriteString("tolerance_file_status", o.Tolerances.Status);
        w.WriteEndObject();

        // Per class: how many cases and how close to the limit the worst element came (1.0 is the limit).
        w.WriteStartObject("classes");
        foreach (IGrouping<string, CaseResult> g in report.Results.Where(r => r.Outputs.Count > 0).GroupBy(r => r.Class).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            w.WriteStartObject(g.Key);
            w.WriteNumber("cases", g.Count());
            Number(w, "max_utilisation", g.Max(r => r.Outputs.Max(x => x.Stat.MaxUtilisation)));
            Number(w, "max_abs", g.Max(r => r.Outputs.Max(x => x.Stat.MaxAbs)));
            w.WriteNumber("violations", g.Sum(r => r.Outputs.Sum(x => x.Stat.Violations + x.Stat.NonFinite)));
            w.WriteEndObject();
        }

        w.WriteEndObject();

        w.WriteStartArray("variants");
        foreach (VariantReport v in report.Variants)
        {
            var mine = report.Results.Where(r => r.Variant == v.Info.Id && r.Op == v.Info.Op).ToList();
            w.WriteStartObject();
            w.WriteString("op", v.Info.Op.ToString());
            w.WriteString("variant", v.Info.Id);
            w.WriteString("flags", v.Info.Flags.ToString());
            w.WriteNumber("passed", mine.Count(r => r.Outcome == "pass"));
            w.WriteNumber("failed", mine.Count(r => r.Outcome is "fail" or "error"));
            w.WriteNumber("not_run", mine.Count(r => r.Outcome == "not_run"));
            w.WriteString("abi_probe", v.AbiProbe);
            w.WriteEndObject();
        }

        w.WriteEndArray();
        w.WriteEndObject();
        w.WriteEndObject();
    }

    /// <summary>One-screen text summary for a console.</summary>
    /// <param name="report">The report.</param>
    /// <returns>Lines.</returns>
    public static IEnumerable<string> Summary(ConformanceReport report)
    {
        yield return $"{report.Backend}: {report.Count("pass")} passed, {report.Count("fail")} failed, {report.Count("error")} errors, {report.Count("not_run")} not run";
        foreach (VariantReport v in report.Variants)
        {
            var mine = report.Results.Where(r => r.Variant == v.Info.Id && r.Op == v.Info.Op).ToList();
            yield return $"  {v.Info.Op,-15} {v.Info.Id,-34} pass={mine.Count(r => r.Outcome == "pass"),-4} fail={mine.Count(r => r.Outcome is "fail" or "error"),-3} not_run={mine.Count(r => r.Outcome == "not_run"),-3} abi={v.AbiProbe}";
        }
    }
}
