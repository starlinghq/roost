using System.Reflection;
using System.Runtime.InteropServices;

namespace Starlings.Roost.Cuda;

/// <summary>The operating systems whose CUDA library names the resolver knows.</summary>
public enum CudaPlatform
{
    /// <summary>Linux (shared objects with a version suffix).</summary>
    Linux,

    /// <summary>Windows (DLLs with the major in the name).</summary>
    Windows,
}

/// <summary>
/// Maps the logical library names of the generated bindings (<see cref="CudaLibraries"/>) to the file names of each
/// operating system. The CUDA major comes from the caller, which takes it from the kernel manifest's
/// <c>toolkit.cuda</c> (build plan section 2.1, decision 17); nothing here hard-codes 12.
/// </summary>
public static class CudaLibraryNames
{
    /// <summary>The file names to try for a logical library, in order.</summary>
    /// <param name="logicalName">One of <see cref="CudaLibraries"/>.</param>
    /// <param name="cudaMajor">The CUDA toolkit major, for example 13.</param>
    /// <param name="platform">The operating system.</param>
    /// <returns>Candidate file names; empty when <paramref name="logicalName"/> is not a CUDA library.</returns>
    public static IReadOnlyList<string> Candidates(string logicalName, int cudaMajor, CudaPlatform platform)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(cudaMajor, 1);
        return (platform, logicalName) switch
        {
            // The runtime and cuBLASLt are versioned by the toolkit major; the driver and NVML ship with the driver.
            (CudaPlatform.Linux, CudaLibraries.Cudart) => [$"libcudart.so.{cudaMajor}", "libcudart.so"],
            (CudaPlatform.Linux, CudaLibraries.Cuda) => ["libcuda.so.1", "libcuda.so"],
            (CudaPlatform.Linux, CudaLibraries.CublasLt) => [$"libcublasLt.so.{cudaMajor}", "libcublasLt.so"],
            (CudaPlatform.Linux, CudaLibraries.Nvml) => ["libnvidia-ml.so.1", "libnvidia-ml.so"],
            (CudaPlatform.Windows, CudaLibraries.Cudart) => [$"cudart64_{cudaMajor}.dll"],
            (CudaPlatform.Windows, CudaLibraries.Cuda) => ["nvcuda.dll"],
            (CudaPlatform.Windows, CudaLibraries.CublasLt) => [$"cublasLt64_{cudaMajor}.dll"],
            (CudaPlatform.Windows, CudaLibraries.Nvml) => ["nvml.dll"],
            _ => [],
        };
    }

    /// <summary>The platform this process runs on.</summary>
    /// <returns>The platform.</returns>
    /// <exception cref="PlatformNotSupportedException">The OS is neither Linux nor Windows.</exception>
    public static CudaPlatform Current() =>
        OperatingSystem.IsLinux() ? CudaPlatform.Linux
        : OperatingSystem.IsWindows() ? CudaPlatform.Windows
        : throw new PlatformNotSupportedException("Roost's CUDA bindings support Linux and Windows.");
}

/// <summary>
/// Installs the <see cref="NativeLibrary"/> resolver of the generated bindings. The bindings import logical names; this
/// resolver loads the first file that exists, trying the OS loader's default search first and then the directories below.
/// </summary>
public static class CudaLibraryResolver
{
    private static readonly Lock Gate = new();
    private static int _installedMajor;

    /// <summary>Installs the resolver once per process. Calling it again with the same major is a no-op.</summary>
    /// <param name="cudaMajor">The CUDA toolkit major, from the kernel manifest's <c>toolkit.cuda</c>.</param>
    /// <exception cref="NotSupportedException">
    /// <paramref name="cudaMajor"/> differs from the toolkit the bindings were generated from
    /// (<see cref="CudaToolkit.Major"/>), or another major is already installed.
    /// </exception>
    public static void Install(int cudaMajor = CudaToolkit.Major)
    {
        if (cudaMajor != CudaToolkit.Major)
        {
            throw new NotSupportedException(
                $"The CUDA bindings were generated from CUDA {CudaToolkit.Major}; the manifest asks for CUDA {cudaMajor}.");
        }

        lock (Gate)
        {
            if (_installedMajor != 0)
            {
                return;
            }

            NativeLibrary.SetDllImportResolver(typeof(CudaLibraryResolver).Assembly, Resolve);
            _installedMajor = cudaMajor;
        }
    }

    /// <summary>Directories searched after the OS loader's default search, most specific first.</summary>
    /// <param name="platform">The operating system.</param>
    /// <param name="environment">Reads an environment variable; null when unset.</param>
    /// <returns>Existing or not: callers test for files.</returns>
    public static IReadOnlyList<string> SearchDirectories(CudaPlatform platform, Func<string, string?> environment)
    {
        ArgumentNullException.ThrowIfNull(environment);
        var dirs = new List<string>();
        void Add(string? dir)
        {
            if (!string.IsNullOrEmpty(dir))
            {
                dirs.Add(dir);
            }
        }

        Add(environment("ROOST_CUDA_LIB_DIR"));
        if (platform == CudaPlatform.Linux)
        {
            foreach (var variable in new[] { "CUDA_HOME", "CUDA_PATH" })
            {
                var root = environment(variable);
                if (!string.IsNullOrEmpty(root))
                {
                    Add(Path.Combine(root, "lib64"));
                    Add(Path.Combine(root, "targets/x86_64-linux/lib"));
                }
            }

            Add("/opt/cuda/lib64");
            Add("/usr/local/cuda/lib64");
            Add("/usr/lib/x86_64-linux-gnu");
            Add("/usr/lib64");
            Add("/usr/lib");
        }
        else
        {
            var root = environment("CUDA_PATH");
            if (!string.IsNullOrEmpty(root))
            {
                Add(Path.Combine(root, "bin", "x64"));
                Add(Path.Combine(root, "bin"));
            }
        }

        return dirs;
    }

    private static nint Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        var platform = CudaLibraryNames.Current();
        var candidates = CudaLibraryNames.Candidates(name, CudaToolkit.Major, platform);
        if (candidates.Count == 0)
        {
            return 0; // not one of ours: default resolution
        }

        foreach (var file in candidates)
        {
            if (NativeLibrary.TryLoad(file, assembly, searchPath, out var handle))
            {
                return handle;
            }
        }

        foreach (var dir in SearchDirectories(platform, Environment.GetEnvironmentVariable))
        {
            foreach (var file in candidates)
            {
                var path = Path.Combine(dir, file);
                if (File.Exists(path) && NativeLibrary.TryLoad(path, out var handle))
                {
                    return handle;
                }
            }
        }

        return 0; // the runtime then throws DllNotFoundException naming the logical library
    }
}
