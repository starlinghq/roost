namespace Starlings.Roost.Engine.Threading;

/// <summary>
/// A dedicated thread that calls a poll function in a loop: the engine thread and the worker thread of the
/// in-process transport are both one of these. Waiting is a bounded spin, then a yield; there is no sleep, because
/// the rings are polled and a sleeping thread adds its wake-up latency to every step.
/// </summary>
/// <remarks>
/// <para>The loop allocates nothing: the delegate and the thread are created once, in the constructor and in
/// <see cref="Start"/>. The stop flag is read with <c>Volatile.Read</c> and written with <c>Volatile.Write</c>; it
/// carries no data, so only visibility (no hoisting of the read out of the loop by the JIT, and prompt publication)
/// matters, not ordering with other memory. A poll function that throws stops the loop and the exception is kept in
/// <see cref="Fault"/>.</para>
/// <para><b>Lifetime.</b> A poll function that reads a <see cref="Rings.RingSegment"/> must not outlive the mapping.
/// Create such a thread with <see cref="Rings.RingSegment.CreatePoller"/>, so that closing the segment stops and
/// joins it before the view is released (the rule is on <see cref="Rings.RingSegment"/>). <see cref="Stop"/> returns
/// whether the thread has ended; only a true result means the poll function will not run again. The stop flag is
/// checked between polls, so a poll function that waits inside itself must return, or check
/// <see cref="StopRequested"/>, for a stop to take effect.</para>
/// </remarks>
public sealed class PollingThread : IDisposable
{
    /// <summary>How long <see cref="Dispose"/> waits for the thread to end.</summary>
    public static readonly TimeSpan DefaultJoinTimeout = TimeSpan.FromSeconds(5);

    private const int Created = 0;
    private const int Started = 1;
    private const int StoppedUnstarted = 2;

    private readonly Func<bool> _poll;
    private readonly Thread _thread;
    private readonly Lock _gate = new();
    private int _stop;
    private int _state;
    private Exception? _fault;

    /// <summary>Creates a thread; call <see cref="Start"/> to run it.</summary>
    /// <param name="name">The thread name.</param>
    /// <param name="poll">Does one unit of work and returns true if it did any, false if it found nothing.</param>
    public PollingThread(string name, Func<bool> poll)
    {
        ArgumentNullException.ThrowIfNull(poll);
        _poll = poll;
        _thread = new Thread(Run) { Name = name, IsBackground = true };
    }

    /// <summary>The exception that stopped the loop, if any.</summary>
    public Exception? Fault => Volatile.Read(ref _fault);

    /// <summary>Whether a stop has been requested. A poll function that waits inside itself checks this to return early.</summary>
    public bool StopRequested => Volatile.Read(ref _stop) != 0;

    /// <summary>Whether the poll function can no longer run: the thread was never started and now cannot be, or it has ended.</summary>
    public bool HasExited => Volatile.Read(ref _state) switch
    {
        StoppedUnstarted => true,
        Started => !_thread.IsAlive,
        _ => false,
    };

    /// <summary>Starts the thread.</summary>
    /// <exception cref="ObjectDisposedException">The thread was stopped before it started.</exception>
    public void Start()
    {
        // _gate orders Start against Stop: Stop sees either Created (and the thread can then never start) or Started
        // with the thread already started, so Stop never joins a thread that has not started.
        lock (_gate)
        {
            if (_state != Created)
            {
                throw new ObjectDisposedException(_thread.Name, "The polling thread was stopped or already started.");
            }

            _thread.Start();
            Volatile.Write(ref _state, Started);
        }
    }

    /// <summary>Asks the loop to stop after the current poll.</summary>
    public void RequestStop() => Volatile.Write(ref _stop, 1);

    /// <summary>Waits for the thread to end.</summary>
    /// <param name="timeout">How long to wait.</param>
    /// <returns>True if it ended.</returns>
    public bool Join(TimeSpan timeout) => _thread.Join(timeout);

    /// <summary>
    /// Requests a stop and waits, up to <paramref name="timeout"/>, for the thread to end. When it returns true, every
    /// memory access the poll function made happened before the return (<c>Thread.Join</c>), and the poll function
    /// will not run again.
    /// </summary>
    /// <param name="timeout">How long to wait.</param>
    /// <returns>True if the thread has ended or never started; false if it is still running.</returns>
    public bool Stop(TimeSpan timeout)
    {
        RequestStop();
        lock (_gate)
        {
            if (_state != Started)
            {
                Volatile.Write(ref _state, StoppedUnstarted);
                return true; // never started, and Start now throws
            }
        }

        return _thread.Join(timeout);
    }

    /// <summary>Stops the thread and waits up to <see cref="DefaultJoinTimeout"/>. Use <see cref="Stop"/> to learn whether it ended.</summary>
    public void Dispose() => _ = Stop(DefaultJoinTimeout);

    private void Run()
    {
        var wait = new SpinWait();
        try
        {
            while (Volatile.Read(ref _stop) == 0)
            {
                if (_poll())
                {
                    wait.Reset();
                }
                else
                {
                    wait.SpinOnce(sleep1Threshold: -1); // spin, then Thread.Yield / Sleep(0); never Sleep(1)
                }
            }
        }
        catch (Exception ex)
        {
            Volatile.Write(ref _fault, ex);
        }
    }
}
