using System.Collections.Immutable;
using System.Runtime.InteropServices;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Eager;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.Worker.Weights;

/// <summary>
/// A model's weights in host memory (the CPU reference backend's "device", <see cref="ModelWeightLoader.LoadHost"/>) that can
/// be released, restored and replaced in place (R-100): the weights half of <see cref="IWeightLifecycle"/> for a CPU executor.
/// Every load goes through the R-024 verified stream, so a file that does not match its lock (size, header, tensors or
/// SHA-256) is refused before any forward can read it.
/// </summary>
/// <remarks>
/// <para><b>Release and restore.</b> <see cref="Release"/> frees the arena and keeps the locked source; <see cref="Restore"/>
/// streams the source again and checks its digest twice: the lock's digest must be the one the host expects (a source the
/// host does not know is refused without being read), and the bytes read must hash to the lock's digest (a file substituted
/// on disk while the worker slept is refused). A failed restore leaves the weights released.</para>
/// <para><b>Replace.</b> <see cref="Replace"/> streams the new file into a second arena and swaps it in only after its digest
/// matched, so a failed replacement leaves the old weights resident and in use; the cost is the two arenas held at once for the
/// length of the load. While the weights are released, the new file is verified the same way and then released, and it becomes
/// the source the next restore loads.</para>
/// <para>Addresses change on every restore and replace: an executor reads <see cref="Weights"/> again at each step. Not
/// thread-safe; the worker's step thread owns it.</para>
/// </remarks>
public sealed class HostWeightResidency : IDisposable
{
    private readonly IReadOnlyList<EagerWeightSlot> _slots;
    private HostModelWeights? _weights;

    /// <summary>Loads and verifies the weights.</summary>
    /// <param name="slots">The model's weight slots.</param>
    /// <param name="path">The safetensors file.</param>
    /// <param name="locked">Its lock entry.</param>
    /// <exception cref="WeightLoadException">The file does not match the lock or the slots.</exception>
    public HostWeightResidency(IReadOnlyList<EagerWeightSlot> slots, string path, LockedFile locked)
    {
        ArgumentNullException.ThrowIfNull(slots);
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(locked);
        _slots = slots;
        _weights = ModelWeightLoader.LoadHost(path, locked, slots);
        SourcePath = path;
        Source = locked;
    }

    /// <summary>The resident weights, or null while released.</summary>
    public HostModelWeights? Weights => _weights;

    /// <summary>Whether the weights are resident.</summary>
    public bool IsResident => _weights is not null;

    /// <summary>The file the next restore loads.</summary>
    public string SourcePath { get; private set; }

    /// <summary>Its lock entry.</summary>
    public LockedFile Source { get; private set; }

    /// <summary>The verified SHA-256 of the resident weights, 32 bytes; default while released.</summary>
    public ImmutableArray<byte> ResidentDigest =>
        _weights is null ? default : ImmutableCollectionsMarshal.AsImmutableArray(Convert.FromHexString(_weights.Verification.Sha256));

    /// <summary>Frees the arena; the source is kept for <see cref="Restore"/>.</summary>
    /// <returns>The outcome, with the bytes freed (0 when already released).</returns>
    public ControlOutcome Release()
    {
        var weights = _weights;
        _weights = null;
        weights?.Dispose();
        return ControlOutcome.Ok(weights?.Bytes ?? 0);
    }

    /// <summary>Makes the weights resident with the expected digest, streaming them again when released.</summary>
    /// <param name="expectedDigest">The SHA-256 the resident weights must have, 32 bytes.</param>
    /// <returns>The outcome: <see cref="ControlStatus.WeightsDigest"/> when the digest or the file does not match (the weights stay as they were).</returns>
    public ControlOutcome Restore(ReadOnlySpan<byte> expectedDigest)
    {
        if (expectedDigest.Length != 32)
        {
            return ControlOutcome.Fail(ControlStatus.Malformed, "The expected weights digest is not 32 bytes.");
        }

        if (_weights is not null)
        {
            return ResidentDigest.AsSpan().SequenceEqual(expectedDigest)
                ? ControlOutcome.Ok(0)
                : ControlOutcome.Fail(ControlStatus.WeightsDigest, $"The resident weights have SHA-256 {_weights.Verification.Sha256}, not the expected {Convert.ToHexStringLower(expectedDigest)}.");
        }

        if (!string.Equals(Source.Sha256, Convert.ToHexStringLower(expectedDigest), StringComparison.Ordinal))
        {
            return ControlOutcome.Fail(ControlStatus.WeightsDigest, $"The weights to restore are locked at SHA-256 {Source.Sha256}, not the expected {Convert.ToHexStringLower(expectedDigest)}.");
        }

        var (loaded, failure) = Load(SourcePath, Source);
        if (loaded is null)
        {
            return failure;
        }

        _weights = loaded;
        return ControlOutcome.Ok(loaded.Bytes);
    }

    /// <summary>Verifies a new weights file and swaps it in (or, while released, makes it the source the next restore loads).</summary>
    /// <param name="source">The new file and its lock.</param>
    /// <returns>The outcome: <see cref="ControlStatus.WeightsDigest"/> when the file does not match its lock; the old weights stay.</returns>
    public ControlOutcome Replace(WeightSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        source.Validate();
        var locked = new LockedFile(Path.GetFileName(source.Path), source.Size, Convert.ToHexStringLower(source.Sha256.AsSpan()));
        var (loaded, failure) = Load(source.Path, locked);
        if (loaded is null)
        {
            return failure;
        }

        long bytes = loaded.Bytes;
        if (_weights is null)
        {
            loaded.Dispose(); // verified; stays released until a restore, which streams and verifies it again
            bytes = 0;
        }
        else
        {
            var old = _weights;
            _weights = loaded;
            old.Dispose();
        }

        SourcePath = source.Path;
        Source = locked;
        return ControlOutcome.Ok(bytes);
    }

    /// <inheritdoc/>
    public void Dispose() => _ = Release();

    private (HostModelWeights? Weights, ControlOutcome Failure) Load(string path, LockedFile locked)
    {
        try
        {
            return (ModelWeightLoader.LoadHost(path, locked, _slots), default);
        }
        catch (WeightLoadException e) when (e.Error.Code == RoostErrorCode.AssetMismatch)
        {
            return (null, ControlOutcome.Fail(ControlStatus.WeightsDigest, e.Message));
        }
        catch (WeightLoadException e)
        {
            return (null, ControlOutcome.Fail(ControlStatus.Failed, e.Message));
        }
    }
}
