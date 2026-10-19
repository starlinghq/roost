using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels;

/// <summary>
/// A kernel library or manifest was refused at startup. <see cref="Error"/> carries the typed C6 error: a digest, ABI or
/// table mismatch is <see cref="RoostErrorCode.AssetMismatch"/>, and a missing SM or specialisation is
/// <see cref="RoostErrorCode.ProfileUnsupported"/> (see <see cref="ProfileUnsupportedException"/>).
/// </summary>
public class KernelLoadException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="code">The typed error code.</param>
    /// <param name="message">What was refused and why; never contains prompt text.</param>
    /// <param name="innerException">The cause, or null.</param>
    public KernelLoadException(RoostErrorCode code, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        Error = new RoostError(code, message);
    }

    /// <summary>The typed error, with its stable code name and HTTP body.</summary>
    public RoostError Error { get; }
}

/// <summary>
/// The loaded kernel libraries have no kernel for an op at a device's SM, or none whose constraints admit the requested
/// dtype, head size, layout or shape. This is a startup failure, never a request failure (build plan section 3.1).
/// </summary>
public sealed class ProfileUnsupportedException : KernelLoadException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="op">The op that cannot run, or <see cref="OpId.None"/> when a whole SM is unavailable.</param>
    /// <param name="sm">The device's SM as major * 10 + minor.</param>
    /// <param name="reason">Why no kernel matched.</param>
    public ProfileUnsupportedException(OpId op, int sm, string reason)
        : base(RoostErrorCode.ProfileUnsupported, Describe(op, sm, reason))
    {
        Op = op;
        Sm = sm;
        Reason = reason;
    }

    /// <summary>The op, or <see cref="OpId.None"/> when no library exists for the SM at all.</summary>
    public OpId Op { get; }

    /// <summary>The device's SM.</summary>
    public int Sm { get; }

    /// <summary>Why no kernel matched.</summary>
    public string Reason { get; }

    private static string Describe(OpId op, int sm, string reason) =>
        op == OpId.None
            ? $"profile unsupported on SM {sm}: {reason}"
            : $"profile unsupported: no kernel for op {op} on SM {sm}: {reason}";
}
