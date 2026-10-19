using Starlings.Roost.Cuda;

namespace Starlings.Roost.Worker.Eager;

/// <summary>
/// The two CUDA streams of the eager worker (R-025): the compute stream runs the forward and the kernel plans; the copy
/// stream carries the plan upload and the logits and result download, so a later step's upload can overlap the end of
/// the previous step once the planner runs ahead (R-032). Both are non-blocking streams. An event orders them: the
/// compute stream waits for the plan upload, and the download waits for the forward.
/// </summary>
public sealed unsafe class WorkerStreams : IDisposable
{
    private readonly int _device;
    private nint _planReady;
    private nint _forwardDone;
    private bool _disposed;

    /// <summary>Creates the streams and events on a device.</summary>
    /// <param name="device">CUDA device ordinal.</param>
    public WorkerStreams(int device)
    {
        _device = device;
        CudaCheck.Runtime(CudaRuntime.cudaSetDevice(device), "cudaSetDevice");
        nint s;
        CudaCheck.Runtime(CudaRuntime.cudaStreamCreateWithFlags(&s, CudaRuntime.cudaStreamNonBlocking), "cudaStreamCreateWithFlags(compute)");
        Compute = s;
        CudaCheck.Runtime(CudaRuntime.cudaStreamCreateWithFlags(&s, CudaRuntime.cudaStreamNonBlocking), "cudaStreamCreateWithFlags(copy)");
        Copy = s;
        nint e;
        CudaCheck.Runtime(CudaRuntime.cudaEventCreateWithFlags(&e, CudaRuntime.cudaEventDisableTiming), "cudaEventCreateWithFlags");
        _planReady = e;
        CudaCheck.Runtime(CudaRuntime.cudaEventCreateWithFlags(&e, CudaRuntime.cudaEventDisableTiming), "cudaEventCreateWithFlags");
        _forwardDone = e;
    }

    /// <summary>The stream that runs kernels.</summary>
    public nint Compute { get; }

    /// <summary>The stream that carries host-device copies.</summary>
    public nint Copy { get; }

    /// <summary>Makes the compute stream wait for everything queued so far on the copy stream (the plan upload).</summary>
    public void ComputeWaitsForCopy()
    {
        CudaCheck.Runtime(CudaRuntime.cudaEventRecord(_planReady, Copy), "cudaEventRecord");
        CudaCheck.Runtime(CudaRuntime.cudaStreamWaitEvent(Compute, _planReady, 0), "cudaStreamWaitEvent");
    }

    /// <summary>Makes the copy stream wait for everything queued so far on the compute stream (the forward).</summary>
    public void CopyWaitsForCompute()
    {
        CudaCheck.Runtime(CudaRuntime.cudaEventRecord(_forwardDone, Compute), "cudaEventRecord");
        CudaCheck.Runtime(CudaRuntime.cudaStreamWaitEvent(Copy, _forwardDone, 0), "cudaStreamWaitEvent");
    }

    /// <summary>Blocks until both streams are idle.</summary>
    public void Synchronize()
    {
        CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(Compute), "cudaStreamSynchronize(compute)");
        CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(Copy), "cudaStreamSynchronize(copy)");
    }

    /// <summary>Destroys the streams and events.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _ = CudaRuntime.cudaSetDevice(_device);
        _ = CudaRuntime.cudaStreamSynchronize(Compute);
        _ = CudaRuntime.cudaStreamSynchronize(Copy);
        _ = CudaRuntime.cudaEventDestroy(_planReady);
        _ = CudaRuntime.cudaEventDestroy(_forwardDone);
        _ = CudaRuntime.cudaStreamDestroy(Compute);
        _ = CudaRuntime.cudaStreamDestroy(Copy);
        _planReady = 0;
        _forwardDone = 0;
    }
}
