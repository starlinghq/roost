using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using Starlings.Roost.Transport;

namespace Starlings.Roost.Engine.Rings;

/// <summary>
/// The C8 shared-memory segment, mapped from a file (<c>/dev/shm</c> where it exists, so the pages are memory-backed)
/// with <c>MAP_SHARED</c> semantics. The host creates it; the worker process opens the same path. The layout is the
/// one in <c>kernels/include/roost_worker_shm.h</c>: a <see cref="SegmentHeader"/> at offset 0 and every other table
/// at a 64-byte aligned byte offset from it. The segment is mapped at a different address in each process, so
/// nothing in it is a pointer.
/// </summary>
/// <remarks>
/// Publication of the segment is not ordered by this class: the creator fills the header before it starts the
/// worker, and starting the process is the synchronisation. The handshake state (R-069) is the later one.
/// </remarks>
public sealed unsafe class RingSegment : IDisposable
{
    /// <summary>Alignment of every table in the segment (<c>ROOST_SHM_ALIGNMENT</c>).</summary>
    public const int Alignment = 64;

    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly SafeMemoryMappedViewHandle _handle;
    private readonly bool _owner;
    private byte* _base;
    private bool _disposed;

    private RingSegment(string path, MemoryMappedFile file, MemoryMappedViewAccessor view, bool owner)
    {
        Path = path;
        _file = file;
        _view = view;
        _owner = owner;
        _handle = view.SafeMemoryMappedViewHandle;
        byte* pointer = null;
        _handle.AcquirePointer(ref pointer);
        _base = pointer + view.PointerOffset;
    }

    /// <summary>The file backing the segment.</summary>
    public string Path { get; }

    /// <summary>The segment's total bytes.</summary>
    public long Bytes => (long)Header.SegmentBytes;

    /// <summary>The segment header (read-only after creation).</summary>
    public ref readonly SegmentHeader Header => ref Unsafe.AsRef<SegmentHeader>(Base);

    internal byte* Base
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _base;
        }
    }

    /// <summary>The default directory for segment files: <c>/dev/shm</c> if present, else the temp directory.</summary>
    /// <returns>A directory path.</returns>
    public static string DefaultDirectory() => Directory.Exists("/dev/shm") ? "/dev/shm" : System.IO.Path.GetTempPath();

    /// <summary>Creates the file, lays the segment out and writes every header. The creator deletes the file on dispose.</summary>
    /// <param name="path">A path that does not exist yet.</param>
    /// <param name="options">The sizes.</param>
    /// <returns>The mapped segment.</returns>
    public static RingSegment Create(string path, SegmentOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var layout = SegmentLayout.Compute(options);
        var stream = new FileStream(path, FileMode.CreateNew, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        try
        {
            stream.SetLength(layout.TotalBytes); // zero filled
            var segment = Map(path, stream, layout.TotalBytes, owner: true);
            layout.WriteHeaders(segment._base);
            return segment;
        }
        catch
        {
            stream.Dispose();
            File.Delete(path);
            throw;
        }
    }

    /// <summary>Maps an existing segment and validates every header and offset before returning.</summary>
    /// <param name="path">The creator's path.</param>
    /// <returns>The mapped segment.</returns>
    /// <exception cref="InvalidDataException">A header, offset or size does not match the C8 layout.</exception>
    public static RingSegment Open(string path)
    {
        var stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        try
        {
            var segment = Map(path, stream, stream.Length, owner: false);
            try
            {
                SegmentLayout.Validate(segment._base, stream.Length);
                return segment;
            }
            catch
            {
                segment.Dispose();
                throw;
            }
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>The host's side of the rings: publishes plans, takes results.</summary>
    /// <returns>An endpoint valid until the segment is disposed.</returns>
    public HostRingEndpoint OpenHost() => new(this);

    /// <summary>The worker's side of the rings: takes plans, publishes results.</summary>
    /// <returns>An endpoint valid until the segment is disposed.</returns>
    public WorkerRingEndpoint OpenWorker() => new(this);

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _base = null;
        _handle.ReleasePointer();
        _view.Dispose();
        _file.Dispose();
        if (_owner)
        {
            try
            {
                File.Delete(Path);
            }
            catch (IOException)
            {
                // Already gone or still open elsewhere; the name is unique per run.
            }
        }
    }

    private static RingSegment Map(string path, FileStream stream, long bytes, bool owner)
    {
        var file = MemoryMappedFile.CreateFromFile(stream, null, bytes, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: false);
        var view = file.CreateViewAccessor(0, bytes, MemoryMappedFileAccess.ReadWrite);
        return new RingSegment(path, file, view, owner);
    }
}
