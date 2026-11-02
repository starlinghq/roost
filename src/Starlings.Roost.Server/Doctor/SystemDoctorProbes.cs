using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;
using Starlings.Generation.Roost;

namespace Starlings.Roost.Server.Doctor;

/// <summary>
/// The probes <c>roost doctor</c> runs on the real machine. No CUDA or NVML library is loaded into this process (decision 21):
/// the driver version comes from <c>/proc/driver/nvidia/version</c> or <c>nvidia-smi</c>, NVML is only looked for on disk, and
/// the GPU list comes from <c>nvidia-smi --query-gpu</c>.
/// </summary>
internal sealed partial class SystemDoctorProbes : IDoctorProbes
{
    private static readonly TimeSpan s_smiTimeout = TimeSpan.FromSeconds(15);

    /// <inheritdoc/>
    public string? DriverVersion()
    {
        if (OperatingSystem.IsLinux() && ReadText("/proc/driver/nvidia/version") is { } text && KernelModuleVersion().Match(text) is { Success: true } m)
        {
            return m.Groups[1].Value;
        }

        var (lines, _) = RunSmi("--query-gpu=driver_version", "--format=csv,noheader");
        return lines?.FirstOrDefault(l => l.Length > 0)?.Trim();
    }

    /// <inheritdoc/>
    public string? FindNvml()
    {
        if (OperatingSystem.IsWindows())
        {
            string system = Environment.GetFolderPath(Environment.SpecialFolder.System);
            string programs = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            return new[] { Path.Combine(system, "nvml.dll"), Path.Combine(programs, "NVIDIA Corporation", "NVSMI", "nvml.dll") }.FirstOrDefault(File.Exists);
        }

        // The loader's search order is not reproduced; these are the directories a driver install, the NVIDIA container toolkit
        // (/usr/local/nvidia) and WSL2 (/usr/lib/wsl/lib) put libnvidia-ml.so.1 in, after LD_LIBRARY_PATH.
        var directories = (Environment.GetEnvironmentVariable("LD_LIBRARY_PATH") ?? string.Empty)
            .Split(':', StringSplitOptions.RemoveEmptyEntries)
            .Concat(["/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib", "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/wsl/lib"]);
        return directories
            .SelectMany(d => new[] { Path.Combine(d, "libnvidia-ml.so.1"), Path.Combine(d, "libnvidia-ml.so") })
            .FirstOrDefault(File.Exists);
    }

    /// <inheritdoc/>
    public GpuListing ListGpus()
    {
        var (lines, error) = RunSmi("--query-gpu=index,name,compute_cap,memory.free", "--format=csv,noheader,nounits");
        if (lines is null)
        {
            return new GpuListing([], error);
        }

        var gpus = new List<DoctorGpu>();
        foreach (var line in lines.Where(l => l.Trim().Length > 0))
        {
            var fields = line.Split(',').Select(f => f.Trim()).ToArray();
            if (fields.Length < 4
                || !int.TryParse(fields[0], NumberStyles.None, CultureInfo.InvariantCulture, out int index)
                || ParseComputeCapability(fields[2]) is not { } sm)
            {
                return new GpuListing(gpus, $"nvidia-smi printed a line this build cannot read: '{line}'");
            }

            long? free = long.TryParse(fields[3], NumberStyles.None, CultureInfo.InvariantCulture, out long mib) ? mib * 1024 * 1024 : null;
            gpus.Add(new DoctorGpu(index, fields[1], sm, free));
        }

        return new GpuListing(gpus);
    }

    /// <inheritdoc/>
    public string? FindPackagedWorker() => RoostWorkerRuntime.FindExecutable();

    /// <inheritdoc/>
    public byte[]? ReadSmallFile(string path)
    {
        try
        {
            return File.Exists(path) ? File.ReadAllBytes(path) : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <inheritdoc/>
    public (string Sha256, long Bytes)? HashFile(string path)
    {
        try
        {
            if (!File.Exists(path))
            {
                return null;
            }

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
            return (Convert.ToHexStringLower(SHA256.HashData(stream)), stream.Length);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> KernelLibraries(string directory) =>
        Directory.Exists(directory) ? [.. Directory.GetFiles(directory, "libroost_kernels_sm*.so").Order(StringComparer.Ordinal)] : [];

    /// <inheritdoc/>
    public string? CheckTokenizer(string path, IReadOnlyList<string> requiredTokens)
    {
        if (!File.Exists(path))
        {
            return $"{path} does not exist";
        }

        Tokenizers.Tokenizer tokenizer;
        try
        {
            tokenizer = Tokenizers.Tokenizer.FromFile(path);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException or FormatException or ArgumentException or InvalidOperationException or NotSupportedException)
        {
            return $"{path} does not load: {e.Message}";
        }

        var missing = requiredTokens.Where(t => tokenizer.TokenToId(t) is null).ToList();
        return missing.Count == 0 ? null : $"{path} lacks the profile's stop token(s) {string.Join(", ", missing)}";
    }

    /// <inheritdoc/>
    public long? FreeDiskBytes(string path)
    {
        try
        {
            string full = Path.GetFullPath(path);
            var drive = DriveInfo.GetDrives()
                .Where(d => d.IsReady && full.StartsWith(d.RootDirectory.FullName, OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
                .MaxBy(d => d.RootDirectory.FullName.Length);
            return drive?.AvailableFreeSpace;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException)
        {
            return null;
        }
    }

    /// <inheritdoc/>
    public long? AvailableMemoryBytes()
    {
        if (!OperatingSystem.IsLinux() || ReadText("/proc/meminfo") is not { } text)
        {
            return null;
        }

        var m = MemAvailable().Match(text);
        return m.Success && long.TryParse(m.Groups[1].Value, NumberStyles.None, CultureInfo.InvariantCulture, out long kib) ? kib * 1024 : null;
    }

    /// <summary>Parses <c>nvidia-smi</c>'s <c>compute_cap</c> (<c>8.6</c>) as major * 10 + minor.</summary>
    /// <param name="text">The field.</param>
    /// <returns>The SM, or null when the field is not <c>major.minor</c>.</returns>
    internal static int? ParseComputeCapability(string text)
    {
        var parts = text.Split('.');
        return parts.Length == 2
            && int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int major)
            && int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int minor)
            && minor < 10
                ? (major * 10) + minor
                : null;
    }

    private static string? ReadText(string path)
    {
        try
        {
            return File.Exists(path) ? File.ReadAllText(path) : null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static (IReadOnlyList<string>? Lines, string? Error) RunSmi(params string[] arguments)
    {
        var start = new ProcessStartInfo("nvidia-smi")
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
        };
        foreach (var a in arguments)
        {
            start.ArgumentList.Add(a);
        }

        try
        {
            using var process = Process.Start(start);
            if (process is null)
            {
                return (null, "nvidia-smi did not start");
            }

            var stdout = process.StandardOutput.ReadToEndAsync();
            var stderr = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(s_smiTimeout))
            {
                process.Kill(entireProcessTree: true);
                return (null, $"nvidia-smi did not answer within {s_smiTimeout.TotalSeconds:0} s");
            }

            if (process.ExitCode != 0)
            {
                return (null, $"nvidia-smi exited with {process.ExitCode}: {stderr.Result.Trim()} {stdout.Result.Trim()}".TrimEnd());
            }

            return (stdout.Result.Split('\n'), null);
        }
        catch (System.ComponentModel.Win32Exception e)
        {
            return (null, $"nvidia-smi could not run ({e.Message})");
        }
    }

    [GeneratedRegex(@"Kernel Module\s+(?:for\s+\S+\s+)?([0-9][0-9.]*)")]
    private static partial Regex KernelModuleVersion();

    [GeneratedRegex(@"^MemAvailable:\s+([0-9]+) kB", RegexOptions.Multiline)]
    private static partial Regex MemAvailable();
}
