namespace Starlings.Roost.Server.Doctor;

/// <summary>One visible GPU.</summary>
/// <param name="Index">The device index, as the driver numbers it.</param>
/// <param name="Name">The product name.</param>
/// <param name="Sm">The compute capability as major * 10 + minor (86 for 8.6).</param>
/// <param name="FreeMemoryBytes">Free device memory, or null when the probe could not read it.</param>
internal sealed record DoctorGpu(int Index, string Name, int Sm, long? FreeMemoryBytes);

/// <summary>What the GPU probe found.</summary>
/// <param name="Gpus">The visible GPUs (empty when none or when listing failed).</param>
/// <param name="Error">Why the GPUs could not be listed, or null when listing ran.</param>
internal sealed record GpuListing(IReadOnlyList<DoctorGpu> Gpus, string? Error = null);

/// <summary>
/// The environment <see cref="RoostDoctor"/> reads, behind one seam so that every check is tested on the CPU with fakes. The
/// system implementation (<see cref="SystemDoctorProbes"/>) loads no CUDA or NVML library into the <c>roost</c> process: it
/// reads files and runs <c>nvidia-smi</c> (decision 21 keeps CUDA in the worker process).
/// </summary>
internal interface IDoctorProbes
{
    /// <summary>The installed NVIDIA driver's version (for example <c>610.57</c>), or null when no driver is found.</summary>
    string? DriverVersion();

    /// <summary>The path of the NVML library, or null when it cannot be found.</summary>
    string? FindNvml();

    /// <summary>Lists the visible GPUs.</summary>
    GpuListing ListGpus();

    /// <summary>The packaged <c>roost-worker</c> executable, or null when there is none beside the application.</summary>
    string? FindPackagedWorker();

    /// <summary>Reads a small file whole, or returns null when it does not exist or cannot be read.</summary>
    byte[]? ReadSmallFile(string path);

    /// <summary>The lower-case hex SHA-256 and length of a file, streamed, or null when it does not exist or cannot be read.</summary>
    (string Sha256, long Bytes)? HashFile(string path);

    /// <summary>The kernel library files (<c>libroost_kernels_sm&lt;N&gt;.so</c>) in a directory; empty when there are none or no directory.</summary>
    IReadOnlyList<string> KernelLibraries(string directory);

    /// <summary>Loads a tokenizer and checks that it holds the given tokens.</summary>
    /// <returns>Null when it loads and holds every token; otherwise why not.</returns>
    string? CheckTokenizer(string path, IReadOnlyList<string> requiredTokens);

    /// <summary>Free bytes on the file system that holds the path, or null when it cannot be read.</summary>
    long? FreeDiskBytes(string path);

    /// <summary>Host memory available to new processes (Linux <c>MemAvailable</c>), or null when it cannot be read.</summary>
    long? AvailableMemoryBytes();
}
