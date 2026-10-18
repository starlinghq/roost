using System.Diagnostics;

namespace Starlings.Roost.WorkerHost.Tests.Support;

/// <summary>
/// What the fault tests do to a worker process, on each OS. Freeze is SIGSTOP on Linux and <c>NtSuspendProcess</c> on
/// Windows (every thread stops, the heartbeat included); Kill is SIGKILL and <c>TerminateProcess</c>.
/// </summary>
internal static class OsProcess
{
    public static bool Freeze(int pid) => OperatingSystem.IsWindows() ? Win32.Suspend(pid) : Libc.Signal(pid, Libc.SigStop);

    public static bool Thaw(int pid) => OperatingSystem.IsWindows() ? Win32.Resume(pid) : Libc.Signal(pid, Libc.SigCont);

    public static bool Kill(int pid) => OperatingSystem.IsWindows() ? Win32.Terminate(pid) : Libc.Signal(pid, Libc.SigKill);

    /// <summary>Whether the process is running; on Linux a zombie (dead, not yet reaped by its parent) is not.</summary>
    public static bool Exists(int pid)
    {
        if (OperatingSystem.IsWindows())
        {
            using var process = Win32.OpenProcess(pid);
            return process is not null;
        }

        try
        {
            var stat = File.ReadAllText($"/proc/{pid}/stat");
            var state = stat[(stat.LastIndexOf(')') + 2)..][0];
            return state != 'Z' && state != 'X';
        }
        catch (IOException)
        {
            return false;
        }
    }

    /// <summary>The one-minute load average where the OS reports it, for failure messages.</summary>
    public static string LoadAverage()
    {
        try
        {
            return OperatingSystem.IsLinux() ? File.ReadAllText("/proc/loadavg").Split(' ')[0] : "n/a";
        }
        catch (IOException)
        {
            return "n/a";
        }
    }
}
