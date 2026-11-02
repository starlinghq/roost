using System.Globalization;
using Starlings.Generation.Roost;

namespace Starlings.Roost.Server.Doctor;

/// <summary>The arguments of <c>roost doctor</c>; every input is optional, and a check whose input is missing is skipped.</summary>
internal sealed record DoctorCommand
{
    /// <summary>The proposed default floor for free disk space where the weights live (R-046 proposal, not a measured target).</summary>
    public const double DefaultMinFreeDiskGib = 2;

    /// <summary>The proposed default floor for available host memory (R-046 proposal, not a measured target).</summary>
    public const double DefaultMinFreeMemoryGib = 2;

    /// <summary>The model profile name; it selects the tokenizer's required stop tokens.</summary>
    public string Model { get; init; } = RoostCli.Qwen35Profile;

    /// <summary>The safetensors weights file, or null to skip the weights check.</summary>
    public string? Weights { get; init; }

    /// <summary>The weights lock's SHA-256 as hex; required with <see cref="Weights"/>.</summary>
    public string? WeightsSha256 { get; init; }

    /// <summary>The <c>tokenizer.json</c>, or null to skip the tokenizer check.</summary>
    public string? Tokenizer { get; init; }

    /// <summary>The kernel library directory, or null for the worker executable's directory.</summary>
    public string? KernelDirectory { get; init; }

    /// <summary>The <c>roost-worker</c> executable, or null for the packaged one.</summary>
    public string? Worker { get; init; }

    /// <summary>The kernel manifest, or null for the one beside the worker executable.</summary>
    public string? Manifest { get; init; }

    /// <summary>Whether an SM the manifest marks untested may serve (as <c>roost serve --allow-untested-sm</c>).</summary>
    public bool AllowUntestedSm { get; init; }

    /// <summary>Skips the driver, NVML, GPU and SM checks and checks every kernel library found (an image build has no GPU).</summary>
    public bool NoGpu { get; init; }

    /// <summary>Free disk space below this many GiB is a warning.</summary>
    public double MinFreeDiskGib { get; init; } = DefaultMinFreeDiskGib;

    /// <summary>Available host memory below this many GiB is a warning.</summary>
    public double MinFreeMemoryGib { get; init; } = DefaultMinFreeMemoryGib;
}

/// <summary>
/// <c>roost doctor</c> (R-046): checks what <c>roost serve</c> needs before it starts a worker, and reports each check with a
/// typed <see cref="DoctorStatus"/>. It reads the environment only through <see cref="IDoctorProbes"/>, so every rule here is
/// tested on the CPU with fake probes. It reports and never repairs; the worker's loader and the C8 handshake remain the gates
/// that refuse a mismatched asset at start.
/// </summary>
internal static class RoostDoctor
{
    private const double Gib = 1024.0 * 1024 * 1024;

    /// <summary>The tokens each profile's tokenizer must hold (the stop tokens <c>roost serve</c> looks up).</summary>
    private static readonly Dictionary<string, string[]> s_stopTokens = new(StringComparer.Ordinal)
    {
        [RoostCli.Qwen35Profile] = ["<|im_end|>", "<|endoftext|>"],
    };

    /// <summary>Runs every check, in <see cref="DoctorChecks.All"/> order.</summary>
    /// <param name="command">The inputs.</param>
    /// <param name="probes">The environment.</param>
    /// <returns>The report.</returns>
    public static DoctorReport Run(DoctorCommand command, IDoctorProbes probes)
    {
        ArgumentNullException.ThrowIfNull(command);
        ArgumentNullException.ThrowIfNull(probes);
        var checks = new List<DoctorCheck>();

        // The manifest is read first because the driver and SM checks compare against it, but it is reported in its own place.
        var (manifest, manifestCheck, workerDirectory) = ReadManifest(command, probes);

        IReadOnlyList<DoctorGpu> gpus = [];
        if (command.NoGpu)
        {
            const string skipped = "--no-gpu";
            checks.Add(new(DoctorChecks.Driver, DoctorStatus.Skipped, skipped));
            checks.Add(new(DoctorChecks.Nvml, DoctorStatus.Skipped, skipped));
            checks.Add(new(DoctorChecks.Gpu, DoctorStatus.Skipped, skipped));
        }
        else
        {
            checks.Add(Driver(probes.DriverVersion(), manifest));
            checks.Add(probes.FindNvml() is { } nvml
                ? new(DoctorChecks.Nvml, DoctorStatus.Pass, nvml)
                : new(DoctorChecks.Nvml, DoctorStatus.Warn, "no NVML library found (libnvidia-ml.so.1 or nvml.dll; it ships with the driver, and a container needs the utility driver capability)"));
            var listing = probes.ListGpus();
            gpus = listing.Gpus;
            checks.Add(Gpu(listing));
        }

        checks.Add(manifestCheck);
        checks.Add(command.NoGpu ? new(DoctorChecks.Sm, DoctorStatus.Skipped, "--no-gpu") : Sm(gpus, manifest, command.AllowUntestedSm));
        checks.Add(Kernels(command, probes, manifest, gpus, workerDirectory));
        checks.Add(Weights(command, probes));
        checks.Add(Tokenizer(command, probes));
        checks.Add(Disk(command, probes));
        checks.Add(Memory(command, probes));
        return new DoctorReport(checks);
    }

    private static (DoctorManifest? Manifest, DoctorCheck Check, string? WorkerDirectory) ReadManifest(DoctorCommand command, IDoctorProbes probes)
    {
        // The same resolution as `roost serve`: --manifest, else the one beside --worker, else the one beside the packaged worker.
        string? worker = command.Worker ?? probes.FindPackagedWorker();
        string? workerDirectory = worker is null ? null : Path.GetDirectoryName(Path.GetFullPath(worker));
        string? path = command.Manifest ?? (workerDirectory is null ? null : Path.Combine(workerDirectory, RoostWorkerRuntime.ManifestFileName));
        if (path is null)
        {
            return (null, new(DoctorChecks.Manifest, DoctorStatus.Fail, $"no {RoostWorkerRuntime.ExecutableName} beside the application and no --manifest or --worker given"), null);
        }

        if (probes.ReadSmallFile(path) is not { } bytes)
        {
            return (null, new(DoctorChecks.Manifest, DoctorStatus.Fail, $"{path} does not exist or cannot be read"), workerDirectory);
        }

        try
        {
            var manifest = DoctorManifest.Parse(bytes);
            return (manifest, new(
                DoctorChecks.Manifest,
                DoctorStatus.Pass,
                $"{path}: sha256 {manifest.Sha256}, ABI {manifest.AbiMajor}.{manifest.AbiMinor}, CUDA {manifest.Cuda}"), workerDirectory);
        }
        catch (FormatException e)
        {
            return (null, new(DoctorChecks.Manifest, DoctorStatus.Fail, $"{path}: {e.Message}"), workerDirectory);
        }
    }

    private static DoctorCheck Driver(string? version, DoctorManifest? manifest)
    {
        if (version is null)
        {
            return new(DoctorChecks.Driver, DoctorStatus.Fail, "no NVIDIA driver found (no /proc/driver/nvidia/version, and nvidia-smi did not answer)");
        }

        if (manifest is null)
        {
            return new(DoctorChecks.Driver, DoctorStatus.Pass, $"driver {version} (no manifest, so its minimum was not compared)");
        }

        if (Major(version) is not { } have || Major(manifest.MinDriver) is not { } need)
        {
            return new(DoctorChecks.Driver, DoctorStatus.Warn, $"driver {version}; cannot compare it with the manifest's minimum {manifest.MinDriver}");
        }

        return have >= need
            ? new(DoctorChecks.Driver, DoctorStatus.Pass, $"driver {version} (manifest minimum {manifest.MinDriver}, CUDA {manifest.Cuda})")
            : new(DoctorChecks.Driver, DoctorStatus.Fail, $"driver {version} is older than the manifest's minimum {manifest.MinDriver} for CUDA {manifest.Cuda}");
    }

    private static DoctorCheck Gpu(GpuListing listing)
    {
        if (listing.Gpus.Count == 0)
        {
            return listing.Error is null
                ? new(DoctorChecks.Gpu, DoctorStatus.Fail, "no GPU is visible")
                : new(DoctorChecks.Gpu, DoctorStatus.Fail, $"cannot list GPUs: {listing.Error}");
        }

        string list = string.Join("; ", listing.Gpus.Select(g =>
            $"GPU {g.Index}: {g.Name}, SM {g.Sm}" + (g.FreeMemoryBytes is { } free ? $", {free / Gib:0.0} GiB free" : string.Empty)));
        return listing.Error is null
            ? new(DoctorChecks.Gpu, DoctorStatus.Pass, list)
            : new(DoctorChecks.Gpu, DoctorStatus.Warn, $"{list}; then {listing.Error}");
    }

    private static DoctorCheck Sm(IReadOnlyList<DoctorGpu> gpus, DoctorManifest? manifest, bool allowUntested)
    {
        if (manifest is null)
        {
            return new(DoctorChecks.Sm, DoctorStatus.Skipped, "no manifest");
        }

        if (gpus.Count == 0)
        {
            return new(DoctorChecks.Sm, DoctorStatus.Skipped, "no GPU listed");
        }

        var worst = DoctorStatus.Pass;
        var parts = new List<string>();
        foreach (int sm in gpus.Select(g => g.Sm).Distinct().Order())
        {
            var (status, text) = manifest.SmStatuses.TryGetValue(sm, out var s)
                ? s switch
                {
                    "supported" => (DoctorStatus.Pass, $"SM {sm} supported"),
                    "experimental" => (DoctorStatus.Warn, $"SM {sm} experimental (ran on hardware; not admitted by the fidelity gates)"),
                    "untested" when allowUntested => (DoctorStatus.Warn, $"SM {sm} untested, allowed by --allow-untested-sm (a run is evidence, not a claim)"),
                    "untested" => (DoctorStatus.Fail, $"SM {sm} untested (no hardware evidence); serving needs --allow-untested-sm"),
                    "unsupported" => (DoctorStatus.Fail, $"SM {sm} unsupported by the manifest"),
                    _ => (DoctorStatus.Fail, $"SM {sm} has status '{s}', which this build does not know"),
                }
                : (DoctorStatus.Fail, $"SM {sm} is not in the manifest (no kernel library is built for it)");
            worst = Worse(worst, status);
            parts.Add(text);
        }

        return new(DoctorChecks.Sm, worst, string.Join("; ", parts));
    }

    private static DoctorCheck Kernels(DoctorCommand command, IDoctorProbes probes, DoctorManifest? manifest, IReadOnlyList<DoctorGpu> gpus, string? workerDirectory)
    {
        string? directory = command.KernelDirectory ?? workerDirectory;
        if (directory is null)
        {
            return new(DoctorChecks.Kernels, DoctorStatus.Fail, "no kernel directory: pass --kernels, or --worker for the directory beside it");
        }

        var libraries = probes.KernelLibraries(directory);
        if (libraries.Count == 0)
        {
            return new(DoctorChecks.Kernels, DoctorStatus.Fail, $"no kernel library (libroost_kernels_sm<N>.so) in {directory}");
        }

        if (manifest is null)
        {
            return new(DoctorChecks.Kernels, DoctorStatus.Fail, $"{libraries.Count} librar(ies) in {directory}, but no manifest to check their digests against");
        }

        // With GPUs listed, the libraries their SMs load; otherwise every library present (an image build checks them all).
        var sms = gpus.Count > 0 && !command.NoGpu
            ? gpus.Select(g => g.Sm).Distinct().Order().ToList()
            : libraries.Select(SmOfLibrary).OfType<int>().Distinct().Order().ToList();
        var worst = DoctorStatus.Pass;
        var parts = new List<string>();
        foreach (int sm in sms)
        {
            string path = Path.Combine(directory, $"libroost_kernels_sm{sm}.so");
            DoctorStatus status;
            string text;
            if (!manifest.LibraryDigests.TryGetValue(sm, out var expected) || expected.Count == 0)
            {
                (status, text) = (DoctorStatus.Fail, $"SM {sm}: the manifest records no library digest");
            }
            else if (probes.HashFile(path) is not { } actual)
            {
                (status, text) = (DoctorStatus.Fail, $"SM {sm}: {path} is missing");
            }
            else if (expected.Count > 1)
            {
                (status, text) = (DoctorStatus.Fail, $"SM {sm}: the manifest records {expected.Count} different digests for one library");
            }
            else if (!expected.Contains(actual.Sha256))
            {
                (status, text) = (DoctorStatus.Fail, $"SM {sm}: {path} has sha256 {actual.Sha256}, the manifest pins {expected.Single()}");
            }
            else
            {
                (status, text) = (DoctorStatus.Pass, $"SM {sm}: {path} sha256 {actual.Sha256} matches the manifest");
            }

            worst = Worse(worst, status);
            parts.Add(text);
        }

        return new(DoctorChecks.Kernels, worst, string.Join("; ", parts));
    }

    private static DoctorCheck Weights(DoctorCommand command, IDoctorProbes probes)
    {
        if (command.Weights is null)
        {
            return new(DoctorChecks.Weights, DoctorStatus.Skipped, "no --weights given");
        }

        if (probes.HashFile(command.Weights) is not { } actual)
        {
            return new(DoctorChecks.Weights, DoctorStatus.Fail, $"{command.Weights} does not exist or cannot be read");
        }

        return string.Equals(actual.Sha256, command.WeightsSha256, StringComparison.OrdinalIgnoreCase)
            ? new(DoctorChecks.Weights, DoctorStatus.Pass, $"{command.Weights}: {actual.Bytes} bytes, sha256 matches the lock")
            : new(DoctorChecks.Weights, DoctorStatus.Fail, $"{command.Weights}: sha256 {actual.Sha256}, the lock pins {command.WeightsSha256?.ToLowerInvariant()}");
    }

    private static DoctorCheck Tokenizer(DoctorCommand command, IDoctorProbes probes)
    {
        if (command.Tokenizer is null)
        {
            return new(DoctorChecks.Tokenizer, DoctorStatus.Skipped, "no --tokenizer given");
        }

        var tokens = s_stopTokens[command.Model];
        return probes.CheckTokenizer(command.Tokenizer, tokens) is { } problem
            ? new(DoctorChecks.Tokenizer, DoctorStatus.Fail, problem)
            : new(DoctorChecks.Tokenizer, DoctorStatus.Pass, $"{command.Tokenizer} loads and holds {string.Join(", ", tokens)}");
    }

    private static DoctorCheck Disk(DoctorCommand command, IDoctorProbes probes)
    {
        string path = command.Weights is { } weights ? Path.GetDirectoryName(Path.GetFullPath(weights)) ?? "." : ".";
        return probes.FreeDiskBytes(path) is not { } free
            ? new(DoctorChecks.Disk, DoctorStatus.Warn, $"cannot read free space for {path}")
            : Floor(DoctorChecks.Disk, free, command.MinFreeDiskGib, $"free on the file system of {path}");
    }

    private static DoctorCheck Memory(DoctorCommand command, IDoctorProbes probes) =>
        probes.AvailableMemoryBytes() is not { } available
            ? new(DoctorChecks.Memory, DoctorStatus.Warn, "cannot read available host memory on this OS")
            : Floor(DoctorChecks.Memory, available, command.MinFreeMemoryGib, "host memory available");

    private static DoctorCheck Floor(string name, long bytes, double floorGib, string what)
    {
        string text = string.Create(CultureInfo.InvariantCulture, $"{bytes / Gib:0.0} GiB {what} (floor {floorGib:0.##} GiB, proposed)");
        return bytes >= floorGib * Gib ? new(name, DoctorStatus.Pass, text) : new(name, DoctorStatus.Warn, text);
    }

    private static int? SmOfLibrary(string path)
    {
        string name = Path.GetFileName(path);
        const string Prefix = "libroost_kernels_sm";
        return name.StartsWith(Prefix, StringComparison.Ordinal) && name.EndsWith(".so", StringComparison.Ordinal)
            && int.TryParse(name.AsSpan(Prefix.Length, name.Length - Prefix.Length - 3), NumberStyles.None, CultureInfo.InvariantCulture, out int sm)
                ? sm
                : null;
    }

    private static int? Major(string version) =>
        int.TryParse(version.Split('.')[0], NumberStyles.None, CultureInfo.InvariantCulture, out int major) ? major : null;

    private static DoctorStatus Worse(DoctorStatus a, DoctorStatus b) => Rank(b) > Rank(a) ? b : a;

    private static int Rank(DoctorStatus status) => status switch
    {
        DoctorStatus.Fail => 3,
        DoctorStatus.Warn => 2,
        DoctorStatus.Pass => 1,
        _ => 0,
    };
}
