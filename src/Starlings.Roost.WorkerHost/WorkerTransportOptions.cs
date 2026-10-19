using System.Collections.Immutable;
using System.Diagnostics;
using Starlings.Roost.Abi;

namespace Starlings.Roost.WorkerHost;

/// <summary>Host-side settings for either transport that are not part of the C8 offer (<c>WorkerLaunch</c>).</summary>
public sealed record WorkerTransportOptions
{
    /// <summary>The kernel library ABI the host requires (C1); the worker's library must have the same major and at least this minor.</summary>
    public uint AbiVersion { get; init; } = KernelAbi.Version;

    /// <summary>How long a worker may take from spawn to answering the handshake: context, registration and weight upload (S8: about 0.8 s for 3.44 GB in page cache; a cold cache is slower).</summary>
    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromSeconds(120);

    /// <summary>How long a clean shutdown may take before the worker is killed.</summary>
    public TimeSpan ShutdownTimeout { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>How long to wait for a killed worker to be reaped.</summary>
    public TimeSpan KillTimeout { get; init; } = TimeSpan.FromSeconds(10);

    /// <summary>
    /// A heartbeat counts as alive when it advanced during the step and its last beat is no older than this at the
    /// moment the deadline is found missed. Must be well above the heartbeat period and a garbage-collection pause.
    /// </summary>
    public TimeSpan HeartbeatLiveness { get; init; } = TimeSpan.FromMilliseconds(50);

    /// <summary>Where segment files are created; <see cref="Starlings.Roost.Engine.Rings.RingSegment.DefaultDirectory"/> when null. Tests use a private directory to check that no file is left behind.</summary>
    public string? SegmentDirectory { get; init; }
}

/// <summary>The command that starts a worker process. The transport appends <see cref="WorkerEntry.SegmentFlag"/> and the segment path.</summary>
public sealed record WorkerCommand
{
    /// <summary>The executable.</summary>
    public required string FileName { get; init; }

    /// <summary>Arguments before the segment flag (for example the assembly path when the file is <c>dotnet</c>, and the worker's own options).</summary>
    public ImmutableArray<string> Arguments { get; init; } = [];

    /// <summary>Environment variables added to the child's environment.</summary>
    public ImmutableDictionary<string, string> Environment { get; init; } = ImmutableDictionary<string, string>.Empty;

    /// <summary>
    /// A command that re-runs the current executable (and, under the <c>dotnet</c> host, the entry assembly) with extra
    /// arguments. Tests and the benchmark use it; their <c>Main</c> or module initializer calls <see cref="WorkerEntry.Run"/>.
    /// </summary>
    /// <param name="extraArguments">Arguments that select the executor.</param>
    /// <returns>The command.</returns>
    public static WorkerCommand ForCurrentProcess(params string[] extraArguments)
    {
        var host = System.Environment.ProcessPath ?? throw new InvalidOperationException("The current process has no path.");
        var arguments = ImmutableArray.CreateBuilder<string>();
        if (Path.GetFileNameWithoutExtension(host).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
        {
            var entry = System.Environment.GetCommandLineArgs() is { Length: > 0 } argv ? argv[0] : null;
            if (string.IsNullOrEmpty(entry))
            {
                throw new InvalidOperationException("The command line has no entry assembly path.");
            }

            arguments.Add(entry);
        }

        arguments.AddRange(extraArguments);
        return new WorkerCommand { FileName = host, Arguments = arguments.ToImmutable() };
    }

    internal ProcessStartInfo ToStartInfo(string segmentPath)
    {
        var info = new ProcessStartInfo(FileName)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = false,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        foreach (var argument in Arguments)
        {
            info.ArgumentList.Add(argument);
        }

        info.ArgumentList.Add(WorkerEntry.SegmentFlag);
        info.ArgumentList.Add(segmentPath);
        // A crashing CUDA process can spend longer writing a dump than the step deadline (S8, condition 3).
        info.Environment["DOTNET_DbgEnableMiniDump"] = "0";
        info.Environment["DOTNET_EnableCrashReport"] = "0";
        foreach (var (key, value) in Environment)
        {
            info.Environment[key] = value;
        }

        return info;
    }
}

/// <summary>The worker did not answer the handshake: it exited first, or the startup timeout passed.</summary>
public sealed class WorkerStartException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="message">What happened, with the worker's last output.</param>
    public WorkerStartException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception.</summary>
    public WorkerStartException()
    {
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="message">What happened.</param>
    /// <param name="innerException">The cause.</param>
    public WorkerStartException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
