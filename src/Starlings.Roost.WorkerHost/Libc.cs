using System.Runtime.InteropServices;

namespace Starlings.Roost.WorkerHost;

/// <summary>The few POSIX calls the worker and the tests need. Linux only; callers guard with <see cref="OperatingSystem.IsLinux"/>. <see cref="Win32"/> is the Windows counterpart.</summary>
internal static partial class Libc
{
    internal const int RlimitCore = 4;
    internal const int SigKill = 9;
    internal const int SigStop = 19;
    internal const int SigCont = 18;

    [StructLayout(LayoutKind.Sequential)]
    internal struct RLimit
    {
        public ulong Current;
        public ulong Maximum;
    }

    [LibraryImport("libc.so.6", EntryPoint = "setrlimit", SetLastError = true)]
    private static partial int SetRLimit(int resource, in RLimit limit);

    [LibraryImport("libc.so.6", EntryPoint = "getppid")]
    private static partial int GetParentPid();

    [LibraryImport("libc.so.6", EntryPoint = "kill", SetLastError = true)]
    private static partial int Kill(int pid, int signal);

    [LibraryImport("libc.so.6", EntryPoint = "_exit")]
    private static partial void ExitNow(int code);

    /// <summary>Disables core dumps for this process (S8 condition 3: a CUDA process core can take longer than the step deadline to write).</summary>
    internal static void DisableCoreDumps()
    {
        if (OperatingSystem.IsLinux())
        {
            _ = SetRLimit(RlimitCore, new RLimit { Current = 0, Maximum = 0 });
        }
    }

    internal static int ParentPid() => OperatingSystem.IsLinux() ? GetParentPid() : 0;

    /// <summary>Sends a signal; false when the process does not exist or may not be signalled.</summary>
    internal static bool Signal(int pid, int signal) => OperatingSystem.IsLinux() && Kill(pid, signal) == 0;

    /// <summary>Ends the process at once without running managed shutdown; used when the host has died.</summary>
    internal static void Exit(int code) => ExitNow(code);
}
