using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using Microsoft.Win32.SafeHandles;

namespace Starlings.Roost.WorkerHost;

/// <summary>
/// The Windows counterparts of what <see cref="Libc"/> does on Linux. Windows has no <c>getppid</c> that survives the
/// parent's death (the recorded parent id stays and can be reused), and no SIGSTOP or SIGKILL. The worker therefore keeps
/// a handle to the host process, which pins the process object so its id cannot be reused, and asks the kernel whether it
/// has ended; the tests freeze a worker with <c>NtSuspendProcess</c>, which stops every thread of the process including
/// the heartbeat, as SIGSTOP does.
/// </summary>
[SupportedOSPlatform("windows")]
internal static partial class Win32
{
    private const uint WaitObject0 = 0;

    [LibraryImport("kernel32.dll", EntryPoint = "WaitForSingleObject")]
    private static partial uint WaitForSingleObject(SafeProcessHandle handle, uint milliseconds);

    [LibraryImport("kernel32.dll", EntryPoint = "TerminateProcess", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool TerminateProcess(nint handle, uint exitCode);

    [LibraryImport("ntdll.dll", EntryPoint = "NtSuspendProcess")]
    private static partial int NtSuspendProcess(SafeProcessHandle handle);

    [LibraryImport("ntdll.dll", EntryPoint = "NtResumeProcess")]
    private static partial int NtResumeProcess(SafeProcessHandle handle);

    [LibraryImport("winmm.dll", EntryPoint = "timeBeginPeriod")]
    private static partial uint TimeBeginPeriod(uint milliseconds);

    [LibraryImport("winmm.dll", EntryPoint = "timeEndPeriod")]
    private static partial uint TimeEndPeriod(uint milliseconds);

    /// <summary>Opens the process, keeping its object alive so the id cannot name another process; null when it has already ended.</summary>
    internal static Process? OpenProcess(int pid)
    {
        try
        {
            var process = Process.GetProcessById(pid);
            if (!HasEnded(process))
            {
                return process;
            }

            process.Dispose();
        }
        catch (ArgumentException)
        {
            // No such process.
        }

        return null;
    }

    /// <summary>Whether the process has ended, without allocating (<see cref="Process.HasExited"/> allocates on Windows).</summary>
    internal static bool HasEnded(Process process) => WaitForSingleObject(process.SafeHandle, 0) == WaitObject0;

    /// <summary>Suspends every thread of the process. False when the process cannot be opened or suspended.</summary>
    internal static bool Suspend(int pid) => WithProcess(pid, static p => NtSuspendProcess(p.SafeHandle) >= 0);

    /// <summary>Resumes a process suspended by <see cref="Suspend"/>.</summary>
    internal static bool Resume(int pid) => WithProcess(pid, static p => NtResumeProcess(p.SafeHandle) >= 0);

    /// <summary>Ends the process at once; false when it does not exist or may not be terminated.</summary>
    internal static bool Terminate(int pid) => WithProcess(pid, static p => TerminateProcess(p.SafeHandle.DangerousGetHandle(), 1));

    /// <summary>Ends this process at once without running managed shutdown; used when the host has died.</summary>
    internal static void ExitNow(int code) => _ = TerminateProcess(-1, (uint)code); // -1 is the current-process pseudo handle

    /// <summary>
    /// Raises the system timer resolution to 1 ms. At the default (15.6 ms) a 1 ms <c>Thread.Sleep</c> sleeps one tick or
    /// more, which stretches the heartbeat period past what the liveness window assumes. Pair with <see cref="EndFineTimer"/>.
    /// </summary>
    internal static void BeginFineTimer() => _ = TimeBeginPeriod(1);

    /// <summary>Undoes <see cref="BeginFineTimer"/>.</summary>
    internal static void EndFineTimer() => _ = TimeEndPeriod(1);

    private static bool WithProcess(int pid, Func<Process, bool> action)
    {
        try
        {
            using var process = Process.GetProcessById(pid);
            return action(process);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or System.ComponentModel.Win32Exception)
        {
            return false;
        }
    }
}
