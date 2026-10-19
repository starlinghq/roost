using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using Starlings.Roost.Engine.Threading;
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
/// <para>Publication of the segment is not ordered by this class: the creator fills the header before it starts the
/// worker, and starting the process is the synchronisation. The handshake state (R-069) is the later one.</para>
/// <para><b>Lifetime rule.</b> The mapping is released only after every thread this segment knows about has ended. A
/// thread is known when it was made with <see cref="CreatePoller"/>. Endpoints, rings and spans hold raw pointers
/// into the view and are valid only while it is mapped; a thread the segment does not know about (a caller's own
/// thread, another process's mapping is separate) must be stopped and joined by its owner before
/// <see cref="Close"/> or <see cref="Dispose"/>. <see cref="Close"/> does, in this order:</para>
/// <list type="number">
/// <item><description>Under <c>_lock</c>, marks the segment closing and copies the poller list. <see cref="CreatePoller"/>
/// takes the same lock and refuses a closing segment, so no poller can be added after the copy.</description></item>
/// <item><description>Calls <see cref="PollingThread.Stop"/> on each poller: <c>Volatile.Write</c> of its stop flag,
/// which the loop reads with <c>Volatile.Read</c> between polls, then <c>Thread.Join</c> with a bounded timeout. A
/// join that returns true means the thread has ended, and everything it did to the mapping happened before the join
/// returned.</description></item>
/// <item><description>Only if every join returned true: <c>Volatile.Write</c> of the unmapped state (after which
/// <see cref="Base"/> and <see cref="Header"/> throw <see cref="ObjectDisposedException"/>), then
/// <c>ReleasePointer</c>, the view and the file. If any join timed out, nothing is released: the view stays mapped
/// (leaked, if it never ends) and <see cref="Close"/> returns false, because an unmap under a running poller is an
/// access violation (SIGSEGV on Linux) that kills the process. A later <see cref="Close"/> tries again.</description></item>
/// </list>
/// </remarks>
public sealed unsafe class RingSegment : IDisposable
{
    /// <summary>Alignment of every table in the segment (<c>ROOST_SHM_ALIGNMENT</c>).</summary>
    public const int Alignment = 64;

    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly SafeMemoryMappedViewHandle _handle;
    private readonly bool _owner;
    private readonly Lock _lock = new();
    private readonly List<PollingThread> _pollers = [];
    private byte* _base;
    private bool _closing;
    private int _unmapped;

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

    /// <summary>Whether the view is still mapped. False only after a <see cref="Close"/> that joined every poller.</summary>
    public bool IsMapped => Volatile.Read(ref _unmapped) == 0;

    internal byte* Base
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _unmapped) != 0, this);
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

    /// <summary>
    /// Creates a polling thread over this segment, not yet started. <see cref="Close"/> stops and joins it before the
    /// view is released; see the lifetime rule on <see cref="RingSegment"/>.
    /// </summary>
    /// <param name="name">The thread name.</param>
    /// <param name="poll">Does one unit of work and returns true if it did any.</param>
    /// <returns>The thread; call <see cref="PollingThread.Start"/> to run it.</returns>
    /// <exception cref="ObjectDisposedException">The segment is closing or closed.</exception>
    public PollingThread CreatePoller(string name, Func<bool> poll)
    {
        var poller = new PollingThread(name, poll);
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_closing, this);
            _pollers.RemoveAll(p => p.HasExited);
            _pollers.Add(poller);
        }

        return poller;
    }

    /// <summary>
    /// Stops and joins every poller made by <see cref="CreatePoller"/>, then releases the mapping. If a poller does not
    /// end within <paramref name="pollerJoinTimeout"/>, the mapping is kept and the result is false.
    /// </summary>
    /// <param name="pollerJoinTimeout">How long to wait for each poller.</param>
    /// <returns>True when the view is released (or already was); false when a poller still runs and the view stays mapped.</returns>
    public bool Close(TimeSpan pollerJoinTimeout)
    {
        PollingThread[] pollers;
        lock (_lock)
        {
            if (Volatile.Read(ref _unmapped) != 0)
            {
                return true;
            }

            _closing = true; // step 1: no poller is added after this copy
            pollers = [.. _pollers];
        }

        var allEnded = true;
        foreach (var poller in pollers)
        {
            allEnded &= poller.Stop(pollerJoinTimeout); // step 2: Volatile.Write of the stop flag, then Thread.Join
        }

        if (!allEnded)
        {
            return false; // step 3 refused: a poller may still read the view, so it stays mapped
        }

        lock (_lock)
        {
            if (Volatile.Read(ref _unmapped) != 0)
            {
                return true;
            }

            Volatile.Write(ref _unmapped, 1); // step 3: Base throws from here on
            _base = null;
            _handle.ReleasePointer();
            _view.Dispose();
            _file.Dispose();
        }

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

        return true;
    }

    /// <summary>
    /// <see cref="Close"/> with <see cref="PollingThread.DefaultJoinTimeout"/>. If a poller does not end in time the
    /// view stays mapped (see <see cref="IsMapped"/>) rather than being released under it.
    /// </summary>
    public void Dispose() => _ = Close(PollingThread.DefaultJoinTimeout);

    private static RingSegment Map(string path, FileStream stream, long bytes, bool owner)
    {
        var file = MemoryMappedFile.CreateFromFile(stream, null, bytes, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: false);
        var view = file.CreateViewAccessor(0, bytes, MemoryMappedFileAccess.ReadWrite);
        return new RingSegment(path, file, view, owner);
    }
}
