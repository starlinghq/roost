using Starlings.Roost.Requests;

namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// A checkpoint was refused while it was being streamed. <see cref="Error"/> carries the typed C6 error: a size,
/// digest, header or tensor mismatch is <see cref="RoostErrorCode.AssetMismatch"/>, and a device that cannot hold the
/// weights is <see cref="RoostErrorCode.InsufficientDeviceMemory"/>. This is a startup failure: no weight is handed
/// to a forward pass before the digest has matched.
/// </summary>
public sealed class WeightLoadException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="code">The typed error code.</param>
    /// <param name="message">What was refused and why; never contains prompt text.</param>
    /// <param name="innerException">The cause, or null.</param>
    public WeightLoadException(RoostErrorCode code, string message, Exception? innerException = null)
        : base(message, innerException)
    {
        Error = new RoostError(code, message);
    }

    /// <summary>The typed error, with its stable code name and HTTP body.</summary>
    public RoostError Error { get; }
}
