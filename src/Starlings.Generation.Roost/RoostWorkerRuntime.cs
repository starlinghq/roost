using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Starlings.Roost.WorkerHost;

namespace Starlings.Generation.Roost;

/// <summary>
/// Where the packaged worker comes from (R-062). The <c>Starlings.Roost.Worker.Runtime.linux-x64</c> package puts the Native
/// AOT <c>roost-worker</c> executable and the kernel manifest it was built against among the application's native runtime
/// assets; the default process transport starts that executable, so CUDA and the kernel library never load in the
/// application's own process (decision 21).
/// </summary>
public sealed record RoostWorkerLaunch
{
    /// <summary>The safetensors weights file the worker loads.</summary>
    public required string WeightsPath { get; init; }

    /// <summary>The weights file's locked size in bytes; its digest is the profile's <see cref="RoostModelProfile.WeightsDigest"/>.</summary>
    public required long WeightsSize { get; init; }

    /// <summary>The model id the worker binds; null means the profile's <see cref="RoostModelProfile.ModelId"/>.</summary>
    public string? Model { get; init; }

    /// <summary>The worker executable; null means the packaged one (<see cref="RoostWorkerRuntime.FindExecutable"/>).</summary>
    public string? Executable { get; init; }

    /// <summary>
    /// The directory holding <c>libroost_kernels_sm&lt;N&gt;.so</c>; null means the worker executable's directory. The runtime
    /// package ships no kernel library until R-060 fetches it by digest, so a package install must set this today; a directory
    /// without a library is refused before the worker starts.
    /// </summary>
    public string? KernelDirectory { get; init; }

    /// <summary>The kernel manifest; null means the one shipped beside the worker executable.</summary>
    public string? ManifestPath { get; init; }

    /// <summary>Whether a GPU whose SM the manifest marks <c>untested</c> may serve (the run is evidence for that SM, not a claim).</summary>
    public bool AllowUntestedSm { get; init; }
}

/// <summary>Finds the packaged <c>roost-worker</c> and builds the command that starts it.</summary>
public static class RoostWorkerRuntime
{
    /// <summary>The worker executable's file name, without an extension.</summary>
    public const string ExecutableName = "roost-worker";

    /// <summary>The kernel manifest's file name beside the executable.</summary>
    public const string ManifestFileName = "roost-kernels.manifest.toml";

    /// <summary>
    /// Finds the packaged worker: beside the application (a RID-specific build or publish) or under
    /// <c>runtimes/&lt;rid&gt;/native/</c> (a portable build).
    /// </summary>
    /// <param name="baseDirectory">Where to look; null means <see cref="AppContext.BaseDirectory"/>.</param>
    /// <returns>The executable's path, or null when it is not there.</returns>
    public static string? FindExecutable(string? baseDirectory = null) =>
        Candidates(baseDirectory ?? AppContext.BaseDirectory).FirstOrDefault(File.Exists);

    /// <summary>The SHA-256 of a kernel manifest file, the value <see cref="RoostModelProfile.ManifestDigest"/> pins.</summary>
    /// <param name="manifestPath">The manifest; null means the one shipped beside the packaged worker.</param>
    /// <returns>The digest.</returns>
    /// <exception cref="FileNotFoundException">There is no manifest at the path, or no packaged worker.</exception>
    public static byte[] ManifestDigest(string? manifestPath = null)
    {
        string path = manifestPath ?? Path.Combine(
            Path.GetDirectoryName(FindExecutable() ?? throw NotFound(AppContext.BaseDirectory))!, ManifestFileName);
        return SHA256.HashData(File.ReadAllBytes(path));
    }

    /// <summary>Builds the command that starts the worker for a profile, passing the profile's capacities to it.</summary>
    /// <param name="profile">The model profile.</param>
    /// <param name="worker">The worker settings; <see cref="RoostWorkerOptions.Launch"/> must be set.</param>
    /// <returns>The command.</returns>
    /// <exception cref="ArgumentException"><see cref="RoostWorkerOptions.Launch"/> is not set, or there is no kernel library to load.</exception>
    /// <exception cref="FileNotFoundException">No worker executable was found.</exception>
    /// <exception cref="UnauthorizedAccessException">The executable is not executable and its mode cannot be changed.</exception>
    public static WorkerCommand CommandFor(RoostModelProfile profile, RoostWorkerOptions worker)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(worker);
        var launch = worker.Launch ?? throw new ArgumentException(
            "The process transport (the default, decision 21) needs RoostWorkerOptions.Launch (the packaged roost-worker) or RoostWorkerOptions.Command.",
            nameof(worker));
        string executable = launch.Executable ?? FindExecutable() ?? throw NotFound(AppContext.BaseDirectory);
        if (!File.Exists(executable))
        {
            throw new FileNotFoundException($"The worker executable {executable} does not exist.", executable);
        }

        // Until the kernel library is fetched by digest (decision 12, R-060), the package ships none: refuse here, before a
        // worker is spawned, rather than let it fail to find a library for its SM.
        string kernels = launch.KernelDirectory ?? Path.GetDirectoryName(Path.GetFullPath(executable))!;
        if (!Directory.Exists(kernels) || Directory.GetFiles(kernels, "libroost_kernels_sm*.so").Length == 0)
        {
            throw new ArgumentException(
                $"No kernel library (libroost_kernels_sm<N>.so) in {kernels}: set RoostWorkerLaunch.KernelDirectory to a kernel build.",
                nameof(worker));
        }

        EnsureExecutable(executable);
        var arguments = new WorkerArguments
        {
            Model = launch.Model ?? profile.ModelId,
            WeightsPath = launch.WeightsPath,
            WeightsSize = launch.WeightsSize,
            WeightsSha256 = Convert.ToHexStringLower(profile.WeightsDigest.AsSpan()),
            ManifestPath = launch.ManifestPath,
            KernelDirectory = launch.KernelDirectory,
            AllowUntested = launch.AllowUntestedSm,
            Device = worker.DeviceOrdinal,
            MaxTokens = profile.MaxTokensPerStep,
            MaxSequences = profile.MaxSequencesPerStep,
            MaxKvBlocks = profile.MaxKvBlocksPerStep ?? profile.AttentionBlocks,
            AttentionBlocks = profile.AttentionBlocks,
            RecurrentSlots = profile.RecurrentSlots,
        }.ToArguments();
        return new WorkerCommand { FileName = executable, Arguments = arguments };
    }

    private static IEnumerable<string> Candidates(string baseDirectory)
    {
        string file = OperatingSystem.IsWindows() ? ExecutableName + ".exe" : ExecutableName;
        yield return Path.Combine(baseDirectory, file);
        yield return Path.Combine(baseDirectory, "runtimes", RuntimeInformation.RuntimeIdentifier, "native", file);
        if (OperatingSystem.IsLinux() && RuntimeInformation.OSArchitecture == Architecture.X64)
        {
            yield return Path.Combine(baseDirectory, "runtimes", "linux-x64", "native", file);
        }
    }

    // NuGet does not always keep the execute bit when it extracts a package, so a packaged worker may arrive without it.
    private static void EnsureExecutable(string path)
    {
        if (OperatingSystem.IsWindows())
        {
            return;
        }

        const UnixFileMode execute = UnixFileMode.UserExecute | UnixFileMode.GroupExecute | UnixFileMode.OtherExecute;
        var mode = File.GetUnixFileMode(path);
        if ((mode & UnixFileMode.UserExecute) == 0)
        {
            File.SetUnixFileMode(path, mode | (execute & ReadToExecute(mode)));
        }
    }

    // Grant execute only where read is granted, so the mode does not widen who can use the file.
    private static UnixFileMode ReadToExecute(UnixFileMode mode) =>
        ((mode & UnixFileMode.UserRead) != 0 ? UnixFileMode.UserExecute : 0)
        | ((mode & UnixFileMode.GroupRead) != 0 ? UnixFileMode.GroupExecute : 0)
        | ((mode & UnixFileMode.OtherRead) != 0 ? UnixFileMode.OtherExecute : 0);

    private static FileNotFoundException NotFound(string baseDirectory) => new(
        $"No {ExecutableName} under {baseDirectory}: add the Starlings.Roost.Worker.Runtime.linux-x64 package, or set RoostWorkerLaunch.Executable.",
        Path.Combine(baseDirectory, ExecutableName));
}
