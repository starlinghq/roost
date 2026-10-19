using Roost.KernelOps.Tests.Cuda;

// Usage: Roost.KernelOps.Tests.Cuda <library> [--filter norm,rope,data,invariance,gemm,attention,real,graph] [--log path] [--checkpoint model.safetensors]
// Run through tools/gpu-lease (kernels/README.md). Exit code 0 when every case passed.
string? library = null, logPath = null;
string checkpoint = Environment.GetEnvironmentVariable("ROOST_QWEN35_SAFETENSORS")
    ?? "/home/mark/Projects/roost-wave3/envs/R-013/hub/model.safetensors-00001-of-00001.safetensors";
// "graph" is opt-in (--filter graph): its first run died with a glibc abort from a harness bug (see kernels/README.md), and it has not been rerun.
var groups = new HashSet<string>(["norm", "rope", "data", "invariance", "gemm", "attention", "real"]);
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--filter": groups = [.. args[++i].Split(',')]; break;
        case "--log": logPath = args[++i]; break;
        case "--checkpoint": checkpoint = args[++i]; break;
        default: library ??= args[i]; break;
    }
}

if (library is null)
{
    Console.Error.WriteLine("usage: Roost.KernelOps.Tests.Cuda <library> [--filter ...] [--log path] [--checkpoint file]");
    return 2;
}

using TextWriter log = logPath is null ? TextWriter.Null : new StreamWriter(logPath);
Cuda.Device(0);
int sm = Cuda.ComputeCapability(0);
nint stream = Cuda.NewStream();
using var lib = new Library(library, 0);
var ctx = new Ctx(lib, stream, log, sm);
ctx.Line($"library {lib.Name} abi 0x{lib.AbiVersion:x} entries {lib.Entries.Count}; device sm_{sm}; {DateTime.UtcNow:O}");
void Group(string name, Action<Ctx> body)
{
    if (!groups.Contains(name))
    {
        return;
    }

    ctx.Line($"--- {name}");
    ctx.Rng = new Rng(name.Aggregate(17, (h, ch) => (h * 31) + ch)); // fixed seed per group: string.GetHashCode is randomised per process
    try
    {
        body(ctx);
    }
    catch (Exception e)
    {
        ctx.Check($"{name}: group completed", false, e.GetType().Name + ": " + e.Message);
        Cuda.ClearError();
    }
}

Group("norm", NormCases.Run);
Group("rope", RopeCases.Run);
Group("data", DataCases.Run);
Group("invariance", Invariance.Run);
Group("gemm", GemmCases.Run);
Group("attention", AttentionCases.Run);
Group("graph", GraphCase.Run);
Group("real", c =>
{
    if (File.Exists(checkpoint))
    {
        RealCases.Run(c, checkpoint);
    }
    else
    {
        c.Line($"real weights skipped: {checkpoint} not found");
    }
});

int failed = ctx.Results.Count(r => !r.Pass);
ctx.Line($"=== {ctx.Results.Count - failed} passed, {failed} failed");
Cuda.DestroyStream(stream);
return failed == 0 ? 0 : 1;
