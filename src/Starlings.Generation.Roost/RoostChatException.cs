using Starlings.Roost.Requests;

namespace Starlings.Generation.Roost;

/// <summary>
/// A request Roost refused or could not finish, with its typed error (contract C6): an unsupported or invalid option
/// (<see cref="RoostErrorCode.InvalidOption"/>, with <see cref="RoostError.Param"/> naming it), a prompt longer than the
/// context, a worker that rejected the handshake, or a device fault.
/// </summary>
public sealed class RoostChatException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="error">The error.</param>
    public RoostChatException(RoostError error)
        : base((error ?? throw new ArgumentNullException(nameof(error))).Message)
    {
        Error = error;
    }

    /// <summary>Creates the exception with an <see cref="RoostErrorCode.Internal"/> error.</summary>
    public RoostChatException()
        : this(new RoostError(RoostErrorCode.Internal, "Internal error."))
    {
    }

    /// <summary>Creates the exception with an <see cref="RoostErrorCode.Internal"/> error.</summary>
    /// <param name="message">What happened.</param>
    public RoostChatException(string message)
        : this(new RoostError(RoostErrorCode.Internal, message))
    {
    }

    /// <summary>Creates the exception with an <see cref="RoostErrorCode.Internal"/> error.</summary>
    /// <param name="message">What happened.</param>
    /// <param name="innerException">The cause.</param>
    public RoostChatException(string message, Exception innerException)
        : base(message, innerException)
    {
        Error = new RoostError(RoostErrorCode.Internal, message);
    }

    /// <summary>Creates the exception.</summary>
    /// <param name="error">The error.</param>
    /// <param name="innerException">The cause.</param>
    public RoostChatException(RoostError error, Exception innerException)
        : base((error ?? throw new ArgumentNullException(nameof(error))).Message, innerException)
    {
        Error = error;
    }

    /// <summary>The typed error; <see cref="RoostError.WriteHttpBody"/> gives its stable HTTP body.</summary>
    public RoostError Error { get; }

    internal static RoostChatException InvalidOption(string param, string message) =>
        new(new RoostError(RoostErrorCode.InvalidOption, message, param));
}
