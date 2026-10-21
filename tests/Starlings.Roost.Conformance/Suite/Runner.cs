using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>Options of one conformance run.</summary>
public sealed class RunOptions
{
    /// <summary>Base seed of the randomized cases (recorded in the results).</summary>
    public ulong BaseSeed { get; init; } = 20261019;

    /// <summary>Which cases to generate.</summary>
    public CaseProfile Profile { get; init; } = CaseProfile.Full;

    /// <summary>The tolerance classes.</summary>
    public required ToleranceTable Tolerances { get; init; }

    /// <summary>Upstream goldens, or null (golden cases are then reported as not run, never as passed).</summary>
    public GoldenSet? Goldens { get; init; }

    /// <summary>Run each case of a deterministic variant twice and require identical bytes.</summary>
    public bool Determinism { get; init; } = true;

    /// <summary>Compare each batch-invariant variant's rows with single-row calls.</summary>
    public bool Invariance { get; init; } = true;

    /// <summary>Only these variants (null for all).</summary>
    public ISet<string>? Variants { get; init; }

    /// <summary>Run only the upstream golden cases (the generated cases compare a backend with CpuOps, which is vacuous for CpuOps itself).</summary>
    public bool GoldenOnly { get; init; }

    /// <summary>Only these ops (null for all).</summary>
    public ISet<OpId>? Ops { get; init; }

    /// <summary>Only cases whose id contains this text (null for all).</summary>
    public string? CaseFilter { get; init; }

    /// <summary>Receives one line per case.</summary>
    public Action<string>? Log { get; init; }

    /// <summary>Combinations a variant must refuse on purpose; a case that matches is expected to return the rule's status from plan, and is neither compared nor counted a pass.</summary>
    public IReadOnlyList<RefusalRule> Refusals { get; init; } = RefusalRules.Registered;
}

/// <summary>Statistics of one output of a case.</summary>
/// <param name="Name">Output name.</param>
/// <param name="Stat">The comparison.</param>
public sealed record OutputResult(string Name, CompareStat Stat);

/// <summary>The result of one case on one variant.</summary>
public sealed class CaseResult
{
    /// <summary>Variant id.</summary>
    public required string Variant { get; init; }

    /// <summary>Case id.</summary>
    public required string Case { get; init; }

    /// <summary>The op.</summary>
    public required OpId Op { get; init; }

    /// <summary>Activation type.</summary>
    public required DType Dtype { get; init; }

    /// <summary>"fixed", "random" or "golden".</summary>
    public required string Source { get; init; }

    /// <summary>Seed the case was drawn from.</summary>
    public ulong Seed { get; init; }

    /// <summary>Tolerance class.</summary>
    public required string Class { get; init; }

    /// <summary>"pass", "fail", "error", "not_run" or "refused" (the variant refused a combination it is registered to refuse, with the typed status).</summary>
    public required string Outcome { get; init; }

    /// <summary>What the outputs were compared with: "CpuOps" or "golden".</summary>
    public string Reference { get; init; } = string.Empty;

    /// <summary>Per-output statistics.</summary>
    public List<OutputResult> Outputs { get; } = [];

    /// <summary>Determinism: "pass", "fail" or "n/a".</summary>
    public string Determinism { get; set; } = "n/a";

    /// <summary>Batch invariance: "pass", "fail" or "n/a".</summary>
    public string Invariance { get; set; } = "n/a";

    /// <summary>Checks the backend made, and the reason for an error or a skip.</summary>
    public List<BackendCheck> Checks { get; } = [];

    /// <summary>Message of an error, or the reason a case was not run.</summary>
    public string Detail { get; set; } = string.Empty;

    /// <summary>Whether the case counts as a pass.</summary>
    public bool Passed => Outcome == "pass";
}

/// <summary>One variant's entry in the report.</summary>
/// <param name="Info">Registration data.</param>
/// <param name="AbiProbe">"pass", "fail" or "n/a".</param>
public sealed record VariantReport(VariantInfo Info, string AbiProbe);

/// <summary>Everything a run produced.</summary>
public sealed class ConformanceReport
{
    /// <summary>The backend's name.</summary>
    public required string Backend { get; init; }

    /// <summary>The options the run used.</summary>
    public required RunOptions Options { get; init; }

    /// <summary>Per case and variant.</summary>
    public List<CaseResult> Results { get; } = [];

    /// <summary>Variants that ran.</summary>
    public List<VariantReport> Variants { get; } = [];

    /// <summary>Whether no case failed or errored. Cases not run do not count as passes or failures.</summary>
    public bool Green => Results.All(r => r.Outcome is "pass" or "not_run" or "refused") && Variants.All(v => v.AbiProbe != "fail");

    /// <summary>Count of results with an outcome.</summary>
    /// <param name="outcome">The outcome.</param>
    /// <returns>The count.</returns>
    public int Count(string outcome) => Results.Count(r => r.Outcome == outcome);
}

/// <summary>Runs the conformance cases of every variant of a backend against the CPU reference or the upstream goldens.</summary>
public static class ConformanceRunner
{
    /// <summary>Runs the suite.</summary>
    /// <param name="candidate">The backend under test.</param>
    /// <param name="options">The options.</param>
    /// <returns>The report.</returns>
    public static ConformanceReport Run(IKernelBackend candidate, RunOptions options)
    {
        var report = new ConformanceReport { Backend = candidate.Name, Options = options };
        using var reference = new CpuBackend();
        foreach (VariantInfo v in candidate.Variants.OrderBy(v => (uint)v.Op).ThenBy(v => v.Id, StringComparer.Ordinal))
        {
            if ((options.Variants is not null && !options.Variants.Contains(v.Id)) || (options.Ops is not null && !options.Ops.Contains(v.Op)))
            {
                continue;
            }

            if (!CaseCatalog.Ops.Contains(v.Op))
            {
                report.Results.Add(new CaseResult
                {
                    Variant = v.Id,
                    Case = "(no cases)",
                    Op = v.Op,
                    Dtype = DType.Undefined,
                    Source = "none",
                    Class = string.Empty,
                    Outcome = "not_run",
                    Detail = $"the suite has no cases for {v.Op}",
                });
                continue;
            }

            string abi = "n/a";
            bool probed = false;
            foreach (OpCase c in Cases(v.Op, options))
            {
                using (c)
                {
                    if (options.CaseFilter is not null && !c.Id.Contains(options.CaseFilter, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    if (!v.Accepts(c, candidate.Sm))
                    {
                        // A case outside the variant's registered constraints is not run: the registration says the variant
                        // does not take it, and the suite never launches what a table excludes.
                        report.Results.Add(NotRun(v, c, "outside the variant's registered constraints"));
                        continue;
                    }

                    if (!probed)
                    {
                        probed = true;
                        int? s = SafeProbe(candidate, v.Id, c);
                        abi = s is null ? "n/a" : (s == (int)KernelStatus.AbiMismatch ? "pass" : "fail");
                    }

                    RefusalRule? rule = options.Refusals.FirstOrDefault(x => x.Variant == v.Id && x.Applies(c));
                    CaseResult r = rule is not null ? RunRefusal(candidate, v, c, rule) : RunCase(candidate, reference, v, c, options);
                    report.Results.Add(r);
                    options.Log?.Invoke($"{r.Outcome.ToUpperInvariant(),-7} {v.Id} {c.Id} {Summary(r)}");
                }
            }

            report.Variants.Add(new VariantReport(v, abi));
            if (options.Goldens is null)
            {
                report.Results.Add(new CaseResult
                {
                    Variant = v.Id,
                    Case = "(goldens)",
                    Op = v.Op,
                    Dtype = DType.Undefined,
                    Source = "golden",
                    Class = string.Empty,
                    Outcome = "not_run",
                    Detail = "no goldens file given; upstream golden cases were not run",
                });
            }
        }

        return report;
    }

    private static IEnumerable<OpCase> Cases(OpId op, RunOptions o)
    {
        if (!o.GoldenOnly)
        {
            foreach (OpCase c in CaseCatalog.Generate(op, o.BaseSeed, o.Profile))
            {
                yield return c;
            }
        }

        if (o.Goldens is not null)
        {
            foreach (OpCase c in o.Goldens.Cases(op))
            {
                yield return c;
            }
        }
    }

    private static int? SafeProbe(IKernelBackend b, string variant, OpCase c)
    {
        try
        {
            return b.ProbeShortStruct(variant, c);
        }
        catch (Exception)
        {
            return -1;
        }
    }

    private static CaseResult NotRun(VariantInfo v, OpCase c, string why) => new()
    {
        Variant = v.Id,
        Case = c.Id,
        Op = c.Op,
        Dtype = c.Dtype,
        Source = c.Source,
        Seed = c.Seed,
        Class = c.ToleranceClass,
        Outcome = "not_run",
        Detail = why,
    };

    private static string Summary(CaseResult r)
    {
        if (r.Outputs.Count == 0)
        {
            return r.Detail;
        }

        double util = r.Outputs.Max(o => o.Stat.MaxUtilisation);
        long bad = r.Outputs.Sum(o => o.Stat.Violations + o.Stat.NonFinite);
        return $"class={r.Class} utilisation={util:F3} violations={bad} det={r.Determinism} inv={r.Invariance} {r.Detail}".TrimEnd();
    }

    private static CaseResult RunCase(IKernelBackend candidate, CpuBackend reference, VariantInfo v, OpCase c, RunOptions o)
    {
        ToleranceClass tol = o.Tolerances[c.ToleranceClass];
        string refName = c.Expected is not null ? "golden" : "CpuOps";
        var r = new CaseResult
        {
            Variant = v.Id,
            Case = c.Id,
            Op = c.Op,
            Dtype = c.Dtype,
            Source = c.Source,
            Seed = c.Seed,
            Class = c.ToleranceClass,
            Outcome = "pass",
            Reference = refName,
        };
        try
        {
            double scale = c.ScaleBuffer is not null ? c.Buffers[c.ScaleBuffer].MaxAbs() : 0;
            using ExecResult got = candidate.Execute(v.Id, c);
            r.Checks.AddRange(got.Checks);
            if (got.Status != 0)
            {
                return Finish(r, "fail", $"the variant refused a case inside its constraints: status {got.Status}: {got.Message}");
            }

            Dictionary<string, HostBuf> want;
            ExecResult? refResult = null;
            if (c.Expected is not null)
            {
                want = c.Expected;
            }
            else
            {
                refResult = reference.Execute(CpuBackend.VariantId, c);
                want = refResult.Outputs;
            }

            try
            {
                foreach (string name in c.Outputs)
                {
                    r.Outputs.Add(new OutputResult(name, Compare.Close(got.Outputs[name], want[name], tol, scale)));
                }
            }
            finally
            {
                refResult?.Dispose();
            }

            bool pass = r.Outputs.All(x => x.Stat.Pass) && r.Checks.All(x => x.Ok);
            if (v.Flags.HasFlag(VariantTraits.Deterministic) && o.Determinism)
            {
                using ExecResult again = candidate.Execute(v.Id, c);
                bool same = again.Status == 0 && c.Outputs.All(n => Compare.BytesEqual(got.Outputs[n], again.Outputs[n]));
                r.Determinism = same ? "pass" : "fail";
                pass &= same;
            }

            if (v.Flags.HasFlag(VariantTraits.BatchInvariant) && o.Invariance && c.SingleRow is not null && c.Rows > 1)
            {
                bool inv = true;
                foreach (int row in new[] { 0, c.Rows - 1, c.Rows / 2 }.Distinct())
                {
                    using OpCase one = c.SingleRow(row);
                    using ExecResult single = candidate.Execute(v.Id, one);
                    inv &= single.Status == 0 && c.Outputs.All(n => RowEqual(got.Outputs[n], single.Outputs[n], row, c.RowWidth[n]));
                }

                r.Invariance = inv ? "pass" : "fail";
                pass &= inv;
            }

            if (c.Op == OpId.PagedAttention && c.IOr("mode", -1) == (int)AttentionMode.Prefill && c.IOr("batch", 0) > 1 && c.Buffers.TryGetValue("q_indptr", out HostBuf? qIndptrBuf))
            {
                // Prefill is not registered batch invariant, but a sequence's output must not depend on who else is in the batch
                // (R-019: the racecheck exception of the FA2 prefill kernel is accepted only with this bit-for-bit evidence).
                int batch = c.I("batch");
                int[] qIndptr = qIndptrBuf.As<int>().ToArray();
                int width = c.I("q_heads") * c.I("head_dim") * Elem.Size(c.Dtype);
                foreach (int seq in new[] { 0, batch - 1, batch / 2 }.Distinct())
                {
                    using OpCase one = CaseCatalog.AttentionSequence(c, seq);
                    using ExecResult alone = candidate.Execute(v.Id, one);
                    bool same = alone.Status == 0
                        && got.Outputs["out"].Raw().Slice(qIndptr[seq] * width, (qIndptr[seq + 1] - qIndptr[seq]) * width).SequenceEqual(alone.Outputs["out"].Raw());
                    r.Checks.Add(new BackendCheck($"prefill batch composition: sequence {seq} alone equals its rows in the batch (bytes)", same, same ? string.Empty : $"status {alone.Status}"));
                    pass &= same;
                }
            }

            return Finish(r, pass ? "pass" : "fail", string.Empty);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return Finish(r, "error", e.GetType().Name + ": " + e.Message);
        }
    }

    private static CaseResult RunRefusal(IKernelBackend candidate, VariantInfo v, OpCase c, RefusalRule rule)
    {
        var r = new CaseResult
        {
            Variant = v.Id,
            Case = c.Id,
            Op = c.Op,
            Dtype = c.Dtype,
            Source = c.Source,
            Seed = c.Seed,
            Class = c.ToleranceClass,
            Outcome = "refused",
            Reference = string.Empty,
        };
        try
        {
            using ExecResult got = candidate.Execute(v.Id, c);
            if (got.Status == (int)rule.Status)
            {
                return Finish(r, "refused", $"{rule.Status} from plan, as registered: {rule.Reason}");
            }

            string what = got.Status == 0 ? "the variant served a combination it is registered to refuse" : $"status {got.Status} ({got.Message}) instead of {rule.Status}";
            return Finish(r, "fail", $"expected a typed refusal: {what}");
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            return Finish(r, "error", e.GetType().Name + ": " + e.Message);
        }
    }

    private static bool RowEqual(HostBuf batch, HostBuf single, int row, int width)
    {
        int es = Elem.Size(batch.Dtype);
        return single.Bytes == (long)width * es
            && batch.Raw().Slice(row * width * es, width * es).SequenceEqual(single.Raw());
    }

    private static CaseResult Finish(CaseResult r, string outcome, string detail)
    {
        var done = new CaseResult
        {
            Variant = r.Variant,
            Case = r.Case,
            Op = r.Op,
            Dtype = r.Dtype,
            Source = r.Source,
            Seed = r.Seed,
            Class = r.Class,
            Outcome = outcome,
            Reference = r.Reference,
            Determinism = r.Determinism,
            Invariance = r.Invariance,
            Detail = detail,
        };
        done.Outputs.AddRange(r.Outputs);
        done.Checks.AddRange(r.Checks);
        return done;
    }
}
