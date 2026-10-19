using System.Runtime.InteropServices;
using Roost.HybridConformance;

// R-034 conformance of the hybrid GPU ops against CpuOps (R-018).
//   Roost.HybridConformance <libroost_kernels_sm*.so> [--only state|gdn|graph] [--log file]
// Run through tools/gpu-lease (the lease sets CUDA_VISIBLE_DEVICES, so the device is ordinal 0).
if (args.Length < 1)
{
    Console.Error.WriteLine("usage: Roost.HybridConformance <library> [--only state|gdn|graph] [--log file]");
    return 2;
}

string? Arg(string name) => Array.IndexOf(args, name) is var at and >= 0 && at + 1 < args.Length ? args[at + 1] : null;

var only = Arg("--only");
var report = new Report();
Cuda.SetDevice(0);
var stream = Cuda.CreateStream();
using var lib = Library.Load(args[0], 0);
report.Line($"library {lib.Name} abi {lib.AbiMajor}.{lib.AbiMinor}; entries: {string.Join(", ", lib.Entries.Select(e => Library.VariantId(e)))}");
report.Check(Verify.ParamsSizes(lib), "registration/params_struct_size", "hybrid entries report the struct sizes of the harness mirrors");
try
{
    if (only is null or "state")
    {
        StateCases.Run(lib, report, stream);
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
        GdnCases.RunGraph(lib, report, stream);
        lib.GraphMode = false;
        report.Prefix = "";
    }
}
finally
{
    Cuda.DestroyStream(stream);
}

report.Line($"{report.Checks} checks, {report.Failures} failed");
report.Save(Arg("--log"));
return report.Failures == 0 ? 0 : 1;

internal static class Verify
{
    public static bool ParamsSizes(Library lib)
    {
        var ok = true;
        void Expect(string variant, int size)
        {
            var e = lib.Find(variant);
            ok &= e is not null && e.Value.ParamsStructSize == (uint)size;
        }

        Expect("causal-conv1d-cuda", Marshal.SizeOf<CausalConv1dParams>());
        Expect("recurrent-state-copy-cuda", Marshal.SizeOf<RecurrentStateCopyParams>());
        Expect("recurrent-state-snapshot-cuda", Marshal.SizeOf<RecurrentStateSnapshotParams>());
        return ok;
    }
}
