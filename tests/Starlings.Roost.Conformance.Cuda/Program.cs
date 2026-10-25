using System.Text.RegularExpressions;
using Starlings.Roost.Abi;
using Starlings.Roost.Conformance;
using Starlings.Roost.Conformance.Cuda;

// Usage:
//   Starlings.Roost.Conformance.Cuda <library> [options]    run every registered variant of a kernel library (GPU half)
//   Starlings.Roost.Conformance.Cuda --cpu [options]         the CPU half: the independent oracle against CpuOps, CpuOps against the goldens
// Options: --profile full|ci|sanitizer  --seed N  --goldens <file>  --lock <file>  --tolerances <file>  --variants a,b  --ops RmsNorm,Gemm
//          --case <text>  --no-determinism  --no-invariance  --results <file|dir>  --commit <40 hex>  --device N  --list-variants
// Run GPU work through tools/gpu-lease (tests/Starlings.Roost.Conformance/README.md). Exit code 0 when every case passed.
string? library = null, goldens = null, lockFile = null, tolerances = null, results = null, commit = null, only = null;
string profile = "full";
ulong seed = 20261019;
int device = 0;
bool cpu = false, det = true, inv = true, list = false;
HashSet<string>? variants = null;
HashSet<OpId>? ops = null;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--cpu": cpu = true; break;
        case "--profile": profile = args[++i]; break;
        case "--seed": seed = ulong.Parse(args[++i]); break;
        case "--goldens": goldens = args[++i]; break;
        case "--lock": lockFile = args[++i]; break;
        case "--tolerances": tolerances = args[++i]; break;
        case "--variants": variants = [.. args[++i].Split(',')]; break;
        case "--ops": ops = [.. args[++i].Split(',').Select(Enum.Parse<OpId>)]; break;
        case "--case": only = args[++i]; break;
        case "--no-determinism": det = false; break;
        case "--no-invariance": inv = false; break;
        case "--results": results = args[++i]; break;
        case "--commit": commit = args[++i]; break;
        case "--device": device = int.Parse(args[++i]); break;
        case "--list-variants": list = true; break;
        default: library ??= args[i]; break;
    }
}

if ((library is null && !cpu) || (library is not null && cpu))
{
    Console.Error.WriteLine("usage: Starlings.Roost.Conformance.Cuda <library> [options] | --cpu [options]  (see the header of Program.cs)");
    return 2;
}

if (commit is not null && !Regex.IsMatch(commit, "^[0-9a-f]{40}$"))
{
    Console.Error.WriteLine("--commit must be 40 lowercase hex digits");
    return 2;
}

ToleranceTable table = tolerances is null ? ToleranceTable.LoadProposal() : ToleranceTable.Load(tolerances);
lockFile ??= Path.Combine(AppContext.BaseDirectory, GoldenLock.FileName);
GoldenSet? golden = goldens is null ? null : GoldenSet.Open(goldens, lockFile);

RunOptions Options(bool goldenOnly) => new()
{
    BaseSeed = seed,
    Profile = CaseProfile.Named(profile),
    Tolerances = table,
    Goldens = golden,
    Determinism = det,
    Invariance = inv,
    Variants = variants,
    Ops = ops,
    CaseFilter = only,
    GoldenOnly = goldenOnly,
    Log = Console.WriteLine,
};

RunEnvironment Env(string stage, string subject, string candidate)
{
    var env = new RunEnvironment { Stage = stage, Subject = subject, Candidate = candidate, RepositoryCommit = commit };
    env.Assets.Add((Path.GetFileName(table.Source), table.Sha256));
    if (golden is not null)
    {
        env.Assets.Add((Path.GetFileName(golden.Path), golden.Sha256));
    }

    env.Options["case_profile"] = profile;
    env.Options["dotnet"] = Environment.Version.ToString();
    return env;
}

if (cpu)
{
    int bad = 0;
    using (var oracle = new OracleBackend())
    {
        ConformanceReport r = ConformanceRunner.Run(oracle, Options(false));
        bad += Finish(r, Env("cpu", "oracle", "Oracle (independent implementation) against CpuOps"), results is null ? null : Path.Combine(results, "oracle.results.json"));
    }

    if (golden is not null)
    {
        using var cpuOps = new CpuBackend();
        ConformanceReport r = ConformanceRunner.Run(cpuOps, Options(true));
        bad += Finish(r, Env("cpu", "cpuops", "CpuOps against the upstream goldens"), results is null ? null : Path.Combine(results, "cpuops-goldens.results.json"));
    }

    return bad == 0 ? 0 : 1;
}

CudaRuntime.Device(device);
(int driver, int runtime) = CudaRuntime.Versions();
using var backend = new CudaBackend(library!, device);
if (list)
{
    foreach (VariantInfo v in backend.Variants.OrderBy(v => (uint)v.Op).ThenBy(v => v.Id, StringComparer.Ordinal))
    {
        Console.WriteLine($"{v.Op}\t{v.Id}\t{v.Flags}");
    }

    return 0;
}

string subject = Regex.Replace(Path.GetFileNameWithoutExtension(library!).Replace('_', '-').ToLowerInvariant(), "^lib", string.Empty);
RunEnvironment cuda = Env("cuda", subject, $"{Path.GetFileName(library!)}@sha256:{RunEnvironment.Digest(library!)}");
cuda.Assets.Add((Path.GetFileName(library!), RunEnvironment.Digest(library!)));
cuda.Options["library_name"] = backend.Name;
cuda.Options["library_abi"] = $"0x{backend.AbiVersion:x}";
cuda.Options["sm"] = backend.Sm.ToString();
cuda.Options["cuda_driver_version"] = driver.ToString();
cuda.Options["cuda_runtime_version"] = runtime.ToString();
cuda.Options["cuda_visible_devices"] = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? string.Empty;
cuda.Options["gpu_lease"] = Environment.GetEnvironmentVariable("ROOST_GPU_LEASE") ?? string.Empty;
Console.WriteLine($"library {backend.Name} abi 0x{backend.AbiVersion:x}, {backend.Variants.Count} variants; device sm_{backend.Sm}; {DateTime.UtcNow:O}");
ConformanceReport report = ConformanceRunner.Run(backend, Options(false));
return Finish(report, cuda, results) == 0 ? 0 : 1;

static int Finish(ConformanceReport report, RunEnvironment env, string? path)
{
    foreach (string line in ResultsWriter.Summary(report))
    {
        Console.WriteLine(line);
    }

    if (path is not null)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        ResultsWriter.Write(report, env, path);
        Console.WriteLine($"results: {path}");
    }

    Console.WriteLine(report.Green ? "=== green" : "=== NOT green");
    return report.Green ? 0 : 1;
}
