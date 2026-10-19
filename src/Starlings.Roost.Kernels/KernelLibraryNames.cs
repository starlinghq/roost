using System.Globalization;
using System.Runtime.InteropServices;

namespace Starlings.Roost.Kernels;

/// <summary>The operating systems whose kernel library file names the loader knows.</summary>
public enum KernelPlatform
{
    /// <summary>Linux: <c>libroost_kernels_sm86.so</c>.</summary>
    Linux,

    /// <summary>Windows: <c>roost_kernels_sm86.dll</c>.</summary>
    Windows,
}

/// <summary>
/// Per-OS file names of the per-SM kernel libraries (kernels/CMakeLists.txt builds one shared library per SM, so a
/// missing SM is a missing file), and a <see cref="NativeLibrary"/>-based resolver over a list of directories.
/// </summary>
public static class KernelLibraryNames
{
    /// <summary>The file name of the library for an SM on an OS.</summary>
    /// <param name="sm">The SM as major * 10 + minor.</param>
    /// <param name="platform">The operating system.</param>
    /// <returns>For example <c>libroost_kernels_sm86.so</c>.</returns>
    public static string FileName(int sm, KernelPlatform platform)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(sm, 1);
        var n = sm.ToString(CultureInfo.InvariantCulture);
        return platform == KernelPlatform.Windows ? $"roost_kernels_sm{n}.dll" : $"libroost_kernels_sm{n}.so";
    }

    /// <summary>The platform this process runs on.</summary>
    /// <returns>The platform.</returns>
    /// <exception cref="PlatformNotSupportedException">The OS is neither Linux nor Windows.</exception>
    public static KernelPlatform Current() =>
        OperatingSystem.IsLinux() ? KernelPlatform.Linux
        : OperatingSystem.IsWindows() ? KernelPlatform.Windows
        : throw new PlatformNotSupportedException("Roost's kernel libraries are built for Linux and Windows.");

    /// <summary>Finds the library file for an SM in a list of directories, first match wins.</summary>
    /// <param name="sm">The SM.</param>
    /// <param name="directories">Directories to search, in order.</param>
    /// <param name="platform">The operating system.</param>
    /// <returns>The full path, or null when no directory holds it.</returns>
    public static string? Find(int sm, IEnumerable<string> directories, KernelPlatform platform)
    {
        ArgumentNullException.ThrowIfNull(directories);
        var name = FileName(sm, platform);
        foreach (var dir in directories)
        {
            var path = Path.Combine(dir, name);
            if (File.Exists(path))
            {
                return Path.GetFullPath(path);
            }
        }

        return null;
    }
}
