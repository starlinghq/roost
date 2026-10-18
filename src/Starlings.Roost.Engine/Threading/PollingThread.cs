namespace Starlings.Roost.Engine.Threading;

/// <summary>
/// A dedicated thread that calls a poll function in a loop: the engine thread and the worker thread of the
/// in-process transport are both one of these. Waiting is a bounded spin, then a yield; there is no sleep, because
/// the rings are polled and a sleeping thread adds its wake-up latency to every step.
/// </summary>
/// <remarks>
/// The loop allocates nothing: the delegate and the thread are created once, in the constructor and in
/// <see cref="Start"/>. The stop flag is read with <c>Volatile.Read</c> and written with <c>Volatile.Write</c>; it
/// carries no data, so only visibility (no hoisting of the read out of the loop by the JIT, and prompt publication)
/// matters, not ordering with other memory. A poll function that throws stops the loop and the exception is kept in
/// <see cref="Fault"/>.
/// </remarks>
public sealed class PollingThread : IDisposable
{
    private readonly Func<bool> _poll;
    private readonly Thread _thread;
    private int _stop;
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

    /// <summary>Starts the thread.</summary>
    public void Start() => _thread.Start();

    /// <summary>Asks the loop to stop after the current poll.</summary>
    public void RequestStop() => Volatile.Write(ref _stop, 1);

    /// <summary>Waits for the thread to end.</summary>
    /// <param name="timeout">How long to wait.</param>
    /// <returns>True if it ended.</returns>
    public bool Join(TimeSpan timeout) => _thread.Join(timeout);

    /// <inheritdoc />
    public void Dispose()
    {
        RequestStop();
        _ = _thread.Join(TimeSpan.FromSeconds(5));
    }

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
