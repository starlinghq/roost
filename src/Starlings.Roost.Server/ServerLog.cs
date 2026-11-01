namespace Starlings.Roost.Server;

/// <summary>The server's log messages (source-generated; no prompt or output text is ever logged).</summary>
internal static partial class ServerLog
{
    [LoggerMessage(EventId = 1, Level = LogLevel.Error, Message = "A request failed with an unexpected exception.")]
    public static partial void Unhandled(ILogger logger, Exception exception);
}
