namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// The destination of a weight stream: one arena, written in pieces as the file is read. The streamer calls
/// <see cref="Allocate"/> once, <see cref="Write"/> many times, and then exactly one of <see cref="Commit"/> (the digest
/// matched) or <see cref="Discard"/> (anything failed). An implementation hands out no address before
/// <see cref="Commit"/>, so a forward pass can never read unverified weights.
/// </summary>
public interface IWeightSink : IDisposable
{
    /// <summary>Reserves the arena.</summary>
    /// <param name="bytes">Arena size.</param>
    /// <exception cref="WeightLoadException">The memory is not available (<c>InsufficientDeviceMemory</c>).</exception>
    void Allocate(long bytes);

    /// <summary>Copies bytes into the arena. The data may be reused as soon as this returns.</summary>
    /// <param name="offset">Arena offset.</param>
    /// <param name="data">The bytes.</param>
    void Write(long offset, ReadOnlySpan<byte> data);

    /// <summary>Waits for every pending copy and makes the arena readable.</summary>
    void Commit();

    /// <summary>Releases the arena after a failed stream. Nothing was handed out.</summary>
    void Discard();
}
