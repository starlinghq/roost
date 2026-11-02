namespace Starlings.Roost.Server.Doctor;

/// <summary>The outcome of one <c>roost doctor</c> check (R-046).</summary>
internal enum DoctorStatus
{
    /// <summary>The check passed.</summary>
    Pass,

    /// <summary>Serving may work, but something is degraded or carries no evidence (an experimental SM, low free memory).</summary>
    Warn,

    /// <summary>Serving will not start, or would serve against the wrong assets.</summary>
    Fail,

    /// <summary>Not checked: its input was not given (no <c>--weights</c>), or <c>--no-gpu</c> was passed.</summary>
    Skipped,
}

/// <summary>One check's typed result.</summary>
/// <param name="Name">The check's stable name (<see cref="DoctorChecks"/>).</param>
/// <param name="Status">The outcome.</param>
/// <param name="Detail">What was found, in one line, with paths, versions and digests.</param>
internal sealed record DoctorCheck(string Name, DoctorStatus Status, string Detail);

/// <summary>Every check of one <c>roost doctor</c> run, in the order they ran.</summary>
/// <param name="Checks">The results.</param>
internal sealed record DoctorReport(IReadOnlyList<DoctorCheck> Checks)
{
    /// <summary>Whether no check failed (warnings and skipped checks do not fail the run).</summary>
    public bool Healthy => Checks.All(c => c.Status != DoctorStatus.Fail);

    /// <summary>The result of the named check.</summary>
    /// <param name="name">A <see cref="DoctorChecks"/> name.</param>
    /// <returns>The result.</returns>
    /// <exception cref="InvalidOperationException">No check of that name ran.</exception>
    public DoctorCheck this[string name] => Checks.Single(c => c.Name == name);

    /// <summary>Writes one line per check, <c>[status] name: detail</c>, then a summary line.</summary>
    /// <param name="output">Where to write.</param>
    public void WriteTo(TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(output);
        foreach (var check in Checks)
        {
            output.WriteLine($"[{Label(check.Status)}] {check.Name}: {check.Detail}");
        }

        int fails = Checks.Count(c => c.Status == DoctorStatus.Fail);
        int warns = Checks.Count(c => c.Status == DoctorStatus.Warn);
        output.WriteLine(Healthy
            ? $"roost doctor: no check failed ({warns} warning(s))."
            : $"roost doctor: {fails} check(s) failed, {warns} warning(s).");
    }

    /// <summary>The lower-case label of a status, as <see cref="WriteTo"/> prints it.</summary>
    /// <param name="status">The status.</param>
    /// <returns><c>pass</c>, <c>warn</c>, <c>fail</c> or <c>skip</c>.</returns>
    public static string Label(DoctorStatus status) => status switch
    {
        DoctorStatus.Pass => "pass",
        DoctorStatus.Warn => "warn",
        DoctorStatus.Fail => "fail",
        _ => "skip",
    };
}

/// <summary>The names of the checks, in the order <see cref="RoostDoctor.Run"/> runs them.</summary>
internal static class DoctorChecks
{
    /// <summary>The NVIDIA driver is installed and at least the manifest's <c>toolkit.min_driver</c>.</summary>
    public const string Driver = "driver";

    /// <summary>The NVML library (it ships with the driver) can be found.</summary>
    public const string Nvml = "nvml";

    /// <summary>At least one GPU is visible, with its SM and free memory.</summary>
    public const string Gpu = "gpu";

    /// <summary>The kernel manifest (contract C5) is found and parses; its SHA-256 is the value a profile pins.</summary>
    public const string Manifest = "manifest";

    /// <summary>Each visible GPU's SM has a library in the manifest and the status that allows serving.</summary>
    public const string Sm = "sm";

    /// <summary>The kernel library for each SM is present and its SHA-256 equals the manifest's.</summary>
    public const string Kernels = "kernels";

    /// <summary>The weights file's SHA-256 equals the weights lock (<c>--weights-sha256</c>).</summary>
    public const string Weights = "weights";

    /// <summary>The tokenizer loads and holds the profile's stop tokens.</summary>
    public const string Tokenizer = "tokenizer";

    /// <summary>Free disk space where the weights live.</summary>
    public const string Disk = "disk";

    /// <summary>Available host memory.</summary>
    public const string Memory = "memory";

    /// <summary>Every check name, in run order.</summary>
    public static readonly IReadOnlyList<string> All = [Driver, Nvml, Gpu, Manifest, Sm, Kernels, Weights, Tokenizer, Disk, Memory];
}
