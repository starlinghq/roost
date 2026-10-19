using Starlings.Roost.Abi;
using System.Runtime.InteropServices;
using Roost.HybridConformance;

// R-034 conformance of the hybrid GPU ops against CpuOps (R-018).
//   Roost.HybridConformance <libroost_kernels_sm*.so> [--only state|gating|gdn|graph] [--log file]
// Run through tools/gpu-lease (the lease sets CUDA_VISIBLE_DEVICES, so the device is ordinal 0).
if (args.Length < 1)
{
    Console.Error.WriteLine("usage: Roost.HybridConformance <library> [--only state|gating|gdn|graph] [--log file]");
    return 2;
}

string? Arg(string name) => Array.IndexOf(args, name) is var at and >= 0 && at + 1 < args.Length ? args[at + 1] : null;

var only = Arg("--only");
var report = new Report();
if (!File.Exists(args[0]))
{
    Console.Error.WriteLine($"no such library: {args[0]}");
    return 2;
}

// An exception escaping Main would abort the process with SIGABRT (a core dump on the maintainer's desktop), so every
// failure is caught and reported as a failed check.
// That includes the device setup (no visible device, a card that cannot create a context), the library load and saving the log.
int Finish(int code)
{
    try
    {
        report.Save(Arg("--log"));
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"could not write the log: {ex.Message}");
        return 1;
    }

    return code;
}

nint stream = 0;
Library? libOrNull = null;
try
{
    Cuda.SetDevice(0);
    stream = Cuda.CreateStream();
    libOrNull = Library.Load(args[0], 0);
}
catch (Exception ex)
{
    report.Check(false, "harness/setup", ex.ToString());
    Console.Error.WriteLine(ex.Message);
    if (stream != 0)
    {
        try { Cuda.DestroyStream(stream); } catch (Exception) { /* already reported */ }
    }

    return Finish(1);
}

using var lib = libOrNull;
report.Line($"library {lib.Name} abi {lib.AbiMajor}.{lib.AbiMinor}; entries: {string.Join(", ", lib.Entries.Select(e => Library.VariantId(e)))}");
report.Check(Verify.ParamsSizes(lib), "registration/params_struct_size", "hybrid entries report the struct sizes of the harness mirrors");
try
{
    if (only is null or "state")
    {
        StateCases.Run(lib, report, stream);
    }

    if (only is null or "gating")
    {
        GatingCases.Run(lib, report, stream);
    }

    if (only is null or "gdn")
    {
        GdnCases.Run(lib, report, stream);
    }

    // GRAPH_CAPTURABLE ops: the same cases with run() captured into a CUDA graph and replayed once.
    if (only is null or "graph")
    {
        lib.GraphMode = true;
        report.Prefix = "graph/";
        StateCases.Run(lib, report, stream);
        GatingCases.Run(lib, report, stream);
        GdnCases.RunGraph(lib, report, stream);
        lib.GraphMode = false;
        report.Prefix = "";
    }
}
catch (Exception ex)
{
    report.Check(false, "harness/exception", ex.ToString());
}
finally
{
    try
    {
        Cuda.DestroyStream(stream);
    }
    catch (Exception ex)
    {
        report.Check(false, "harness/destroy-stream", ex.Message);
    }
}

report.Line($"{report.Checks} checks, {report.Failures} failed");
return Finish(report.Failures == 0 ? 0 : 1);

internal static class Verify
{
    public static bool ParamsSizes(Library lib)
    {
        var ok = true;
        void Expect(string variant, int size, int adr)
        {
            var e = lib.Find(variant);
            ok &= e is not null && e.Value.ParamsStructSize == (uint)size && size == adr;
        }

        // Sizes pinned by the ADR and the abi-layout golden: 200, 72, 96 (lifted unchanged from R-034's draft) and 168, 120, 88.
        Expect("causal-conv1d-cuda", Marshal.SizeOf<CausalConv1dParams>(), 200);
        Expect("recurrent-state-copy-cuda", Marshal.SizeOf<RecurrentStateCopyParams>(), 72);
        Expect("recurrent-state-snapshot-cuda", Marshal.SizeOf<RecurrentStateSnapshotParams>(), 96);
        Expect("gated-delta-rule-gates-cuda", Marshal.SizeOf<GatedDeltaRuleGatesParams>(), 168);
        Expect("gated-rms-norm-cuda", Marshal.SizeOf<GatedRmsNormParams>(), 120);
        Expect("sigmoid-mul-cuda", Marshal.SizeOf<SigmoidMulParams>(), 88);
        return ok;
    }
}
