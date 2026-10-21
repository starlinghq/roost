using System.Collections.Immutable;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Kernels;
using Starlings.Roost.Plans;
using Starlings.Roost.Requests;
using Starlings.Roost.Transport;
using Starlings.Roost.Worker.Weights;
using Starlings.Roost.WorkerHost;

namespace Starlings.Roost.Worker.Eager;

/// <summary>
/// The eager worker as the worker process host runs it (<see cref="IStepExecutor"/>, R-069 and R-025): the handshake
/// steps open the device and stream the weights, and each step reads a C2 plan, runs <see cref="EagerWorker"/> and
/// writes the C2 result. A plan the worker refuses and a kernel that returns a status are recoverable faults; a CUDA
/// error is probed (a memset and a sync after <c>cudaGetLastError</c>) and is sticky only when the probe fails too
/// (S8 condition 2).
/// </summary>
public sealed class EagerStepExecutor : IStepExecutor
{
    private readonly IEagerModel _model;
    private readonly EagerWorkerOptions _options;
    private readonly KernelManifest _manifest;
    private readonly string _weightsPath;
    private readonly LockedFile _locked;
    private EagerWorker? _worker;
    private DeviceBuffer? _probe;
    private nint _registered;

    /// <summary>Creates the executor; nothing touches the device until <see cref="OpenDevice"/>.</summary>
    /// <param name="model">The model.</param>
    /// <param name="options">Capacities and locations; <c>Device</c> is replaced by the ordinal the host asks for.</param>
    /// <param name="manifest">The kernel manifest.</param>
    /// <param name="weightsPath">The locked safetensors file.</param>
    /// <param name="locked">Its lock entry.</param>
    public EagerStepExecutor(IEagerModel model, EagerWorkerOptions options, KernelManifest manifest, string weightsPath, LockedFile locked)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(locked);
        _model = model;
        _options = options;
        _manifest = manifest;
        _weightsPath = weightsPath;
        _locked = locked;
    }

    /// <summary>The worker, after <see cref="OpenDevice"/> succeeded (diagnostics and tests).</summary>
    public EagerWorker? Worker => _worker;

    /// <inheritdoc/>
    public WorkerIdentity Identity => new(
        KernelAbi.Version,
        ImmutableCollectionsMarshal.AsImmutableArray(Convert.FromHexString(_manifest.Sha256)),
        ImmutableCollectionsMarshal.AsImmutableArray(Convert.FromHexString(_locked.Sha256)));

    /// <inheritdoc/>
    /// <remarks>The device identity is a digest of the PCI bus id: the CUDA bindings carry no device UUID call yet.</remarks>
    public unsafe HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid)
    {
        try
        {
            _worker = new EagerWorker(_model, _options with { Device = deviceOrdinal }, _manifest);
            _probe = _worker.Allocator.Allocate(256, "probe");
            var bus = new byte[64];
            fixed (byte* p = bus)
            {
                CudaCheck.Runtime(CudaRuntime.cudaDeviceGetPCIBusId(p, bus.Length, deviceOrdinal), "cudaDeviceGetPCIBusId");
            }

            SHA256.HashData(bus.AsSpan(0, Array.IndexOf(bus, (byte)0)))[..Math.Min(16, deviceUuid.Length)].CopyTo(deviceUuid);
            return HandshakeReject.None;
        }
        catch (Exception e) when (e is CudaException or KernelLoadException or DeviceMemoryException or DllNotFoundException)
        {
            _worker?.Dispose();
            _worker = null;
            return HandshakeReject.Device;
        }
    }

    /// <inheritdoc/>
    public unsafe void RegisterSegment(nint address, long bytes)
    {
        CudaCheck.Runtime(CudaRuntime.cudaHostRegister((void*)address, (nuint)bytes, CudaRuntime.cudaHostRegisterPortable), "cudaHostRegister");
        _registered = address;
    }

    /// <inheritdoc/>
    public HandshakeReject LoadWeights()
    {
        try
        {
            (_worker ?? throw new InvalidOperationException("OpenDevice has not run")).LoadWeights(_weightsPath, _locked);
            return HandshakeReject.None;
        }
        catch (WeightLoadException e) when (e.Error.Code == RoostErrorCode.AssetMismatch)
        {
            return HandshakeReject.WeightsDigest;
        }
    }

    /// <inheritdoc/>
    public StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result)
    {
        var worker = _worker ?? throw new InvalidOperationException("OpenDevice has not run");
        try
        {
            var parsed = StepPlanReader.Parse(plan, worker.Limits);
            var output = worker.Step(parsed);
            int bytes = StepResultWriter.Write(result, parsed, output.Tokens, output.Timings);
            return StepOutcome.Ok(bytes);
        }
        catch (PlanRejectedException e)
        {
            return StepOutcome.Recoverable((int)e.Reason);
        }
        catch (KernelRunException e) when (e.Status != KernelStatus.CudaError)
        {
            return StepOutcome.Recoverable((int)e.Status);
        }
        catch (ArgumentException)
        {
            return StepOutcome.Recoverable((int)PlanRejection.TooLarge);
        }
        catch (Exception e) when (e is CudaException or KernelRunException)
        {
            return Probe(e is CudaException c ? c.Status : (int)KernelStatus.CudaError);
        }
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        if (_registered != 0)
        {
            _ = CudaRuntime.cudaHostUnregister((void*)_registered);
            _registered = 0;
        }

        _probe?.Dispose();
        _worker?.Dispose();
        _worker = null;
    }

    private unsafe StepOutcome Probe(int code)
    {
        _ = CudaRuntime.cudaGetLastError();
        int status = CudaRuntime.cudaMemset((void*)_probe!.Address, 0, 256);
        if (status == CudaRuntime.cudaSuccess)
        {
            status = CudaRuntime.cudaDeviceSynchronize();
        }

        return status == CudaRuntime.cudaSuccess ? StepOutcome.Recoverable(code) : StepOutcome.Sticky(code);
    }
}
