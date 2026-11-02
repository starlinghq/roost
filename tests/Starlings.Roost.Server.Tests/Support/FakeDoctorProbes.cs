using System.Security.Cryptography;
using System.Text;
using Starlings.Generation.Roost;
using Starlings.Roost.Server.Doctor;

namespace Starlings.Roost.Server.Tests.Support;

/// <summary>
/// A machine for <c>roost doctor</c> that exists only in memory. By default it is healthy: a recent driver, NVML, one SM 90 GPU
/// whose SM the manifest marks supported, the manifest and the SM 90 library beside the worker with matching digests, ample disk
/// and memory. Tests change one thing each.
/// </summary>
internal sealed class FakeDoctorProbes : IDoctorProbes
{
    private const long Gib = 1024L * 1024 * 1024;

    public FakeDoctorProbes()
    {
        WorkerDirectory = Path.GetDirectoryName(Path.GetFullPath(Worker))!;
        Files[ManifestPath] = Encoding.UTF8.GetBytes(Manifest(new Dictionary<int, string> { [86] = "experimental", [89] = "untested", [90] = "supported" }, new Dictionary<int, byte[]> { [86] = Library86, [90] = Library90 }));
        Files[LibraryPath(90)] = Library90;
        Files[LibraryPath(86)] = Library86;
        Files[WeightsPath] = Weights;
    }

    /// <summary>The bytes of the fake SM 90 kernel library.</summary>
    public static byte[] Library90 { get; } = Encoding.UTF8.GetBytes("fake libroost_kernels_sm90.so");

    /// <summary>The bytes of the fake SM 86 kernel library.</summary>
    public static byte[] Library86 { get; } = Encoding.UTF8.GetBytes("fake libroost_kernels_sm86.so");

    /// <summary>The bytes of the fake weights.</summary>
    public static byte[] Weights { get; } = Encoding.UTF8.GetBytes("fake safetensors");

    public static string WeightsSha256 => Sha(Weights);

    public static string WeightsPath { get; } = Path.GetFullPath(Path.Combine("models", "model.safetensors"));

    public string Worker { get; } = Path.GetFullPath(Path.Combine("opt", "roost", RoostWorkerRuntime.ExecutableName));

    public string WorkerDirectory { get; }

    public string ManifestPath => Path.Combine(WorkerDirectory, RoostWorkerRuntime.ManifestFileName);

    public Dictionary<string, byte[]> Files { get; } = new(StringComparer.Ordinal);

    public string? Driver { get; set; } = "610.57";

    public string? Nvml { get; set; } = "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1";

    public GpuListing Gpus { get; set; } = new([new DoctorGpu(0, "NVIDIA H100", 90, 70 * Gib)]);

    public bool HasPackagedWorker { get; set; } = true;

    public string? TokenizerProblem { get; set; }

    public List<string> TokenizerChecks { get; } = [];

    public long? FreeDisk { get; set; } = 100 * Gib;

    public long? AvailableMemory { get; set; } = 32 * Gib;

    public string LibraryPath(int sm) => Path.Combine(WorkerDirectory, $"libroost_kernels_sm{sm}.so");

    public static string Sha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    /// <summary>A C5-shaped manifest with the given SM statuses and one variant whose per-SM library digests are those of the given bytes.</summary>
    public static string Manifest(IReadOnlyDictionary<int, string> sms, IReadOnlyDictionary<int, byte[]> libraries, string minDriver = "580")
    {
        var text = new StringBuilder();
        text.Append("schema_version = 1\ncontract_version = 3\n\n[abi]\nmajor = 1\nminor = 2\n\n");
        text.Append("[toolkit]\ncuda = \"13.3\"\nmin_driver = \"").Append(minDriver).Append("\"\n\n");
        foreach (var (arch, status) in sms)
        {
            text.Append("[[sm]]\narch = ").Append(arch).Append("\nstatus = \"").Append(status).Append("\"\n\n");
        }

        text.Append("[[ops]]\nop = \"Residual\"\n\n[[ops.variants]]\nid = \"residual-stub\"\nsource = \"roost\"\n\n");
        foreach (var (arch, bytes) in libraries)
        {
            text.Append("[[ops.variants.binaries]]\narch = ").Append(arch).Append("\nsha256 = \"").Append(Sha(bytes)).Append("\"\nbytes = ").Append(bytes.Length).Append("\n\n");
        }

        return text.ToString();
    }

    public string? DriverVersion() => Driver;

    public string? FindNvml() => Nvml;

    public GpuListing ListGpus() => Gpus;

    public string? FindPackagedWorker() => HasPackagedWorker ? Worker : null;

    public byte[]? ReadSmallFile(string path) => Files.GetValueOrDefault(path);

    public (string Sha256, long Bytes)? HashFile(string path) => Files.TryGetValue(path, out var bytes) ? (Sha(bytes), bytes.Length) : null;

    public IReadOnlyList<string> KernelLibraries(string directory) =>
        [.. Files.Keys.Where(p => Path.GetDirectoryName(p) == directory && Path.GetFileName(p).StartsWith("libroost_kernels_sm", StringComparison.Ordinal)).Order(StringComparer.Ordinal)];

    public string? CheckTokenizer(string path, IReadOnlyList<string> requiredTokens)
    {
        TokenizerChecks.Add(path + ": " + string.Join(" ", requiredTokens));
        return TokenizerProblem;
    }

    public long? FreeDiskBytes(string path) => FreeDisk;

    public long? AvailableMemoryBytes() => AvailableMemory;
}
