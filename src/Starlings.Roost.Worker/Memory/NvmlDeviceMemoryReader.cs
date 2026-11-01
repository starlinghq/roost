using System.Globalization;
using Starlings.Roost.Cuda;
using Starlings.Roost.Engine.Admission;

namespace Starlings.Roost.Worker.Memory;

/// <summary>
/// Reads one device's total and free memory through NVML (R-044), for <see cref="PolledDeviceMemorySource"/>. Free memory
/// is the device's, so it sees co-tenants: other processes, other Roost workers, a desktop. Every failure is a typed
/// <see cref="DeviceMemoryReading"/>; nothing throws.
/// </summary>
/// <remarks>
/// <para><b>Lifecycle.</b> NVML is initialised on the first read, not in the constructor, so the reader can be built on a
/// host without a driver. A missing library (<see cref="DeviceMemoryReadStatus.LibraryMissing"/>) is remembered and not
/// probed again; a failed <c>nvmlInit</c> for another reason (driver not loaded yet, a library and kernel driver
/// mismatch) is retried on the next read. A lost device drops its handle so the next read looks it up again, which is how
/// a device that comes back after a reset is read again.</para>
/// <para><b>Which device.</b> Prefer <see cref="ForUuid"/>: NVML indexes devices in PCI bus order, while CUDA ordinals
/// follow <c>CUDA_DEVICE_ORDER</c> (fastest first by default), so an index matches a CUDA ordinal only under
/// <c>CUDA_DEVICE_ORDER=PCI_BUS_ID</c>.</para>
/// <para><b>Which call.</b> <c>nvmlDeviceGetMemoryInfo_v2</c>, whose free bytes leave out memory the driver reserves, and
/// <c>nvmlDeviceGetMemoryInfo</c> on a driver that lacks it.</para>
/// <para><b>Threads.</b> Calls are serialised; <see cref="PolledDeviceMemorySource"/> never overlaps them anyway.</para>
/// </remarks>
public sealed class NvmlDeviceMemoryReader : IDeviceMemoryReader, IDisposable
{
    private readonly INvmlApi _api;
    private readonly uint? _index;
    private readonly string? _uuid;
    private readonly Lock _gate = new();
    private bool _initialised;
    private bool _libraryMissing;
    private bool _useV1;
    private nint _device;
    private bool _disposed;

    private NvmlDeviceMemoryReader(INvmlApi api, uint? index, string? uuid)
    {
        ArgumentNullException.ThrowIfNull(api);
        _api = api;
        _index = index;
        _uuid = uuid;
    }

    /// <summary>A reader of the device with this NVML index.</summary>
    /// <param name="index">The NVML index (PCI bus order).</param>
    /// <param name="api">The NVML calls; <see cref="NvmlApi.Instance"/> when null.</param>
    /// <returns>The reader.</returns>
    public static NvmlDeviceMemoryReader ForIndex(uint index, INvmlApi? api = null) => new(api ?? NvmlApi.Instance, index, null);

    /// <summary>A reader of the device with this UUID (as <c>nvidia-smi -L</c> and <c>cudaDeviceProp.uuid</c> give it).</summary>
    /// <param name="uuid">The UUID, for example <c>GPU-8f3c…</c>.</param>
    /// <param name="api">The NVML calls; <see cref="NvmlApi.Instance"/> when null.</param>
    /// <returns>The reader.</returns>
    public static NvmlDeviceMemoryReader ForUuid(string uuid, INvmlApi? api = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(uuid);
        return new(api ?? NvmlApi.Instance, null, uuid);
    }

    /// <inheritdoc />
    public DeviceMemoryReading Read()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.Failed, "The NVML reader is disposed.");
            }

            if (_libraryMissing)
            {
                return DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.LibraryMissing, "NVML could not be loaded.");
            }

            if (!_initialised)
            {
                var init = _api.Init();
                if (init != NvmlReturn.Success)
                {
                    _libraryMissing = init == NvmlReturn.LibraryNotFound;
                    return Unreadable(ClassifyInit(init), "nvmlInit", init);
                }

                _initialised = true;
            }

            if (_device == 0)
            {
                var found = _uuid is not null ? _api.DeviceGetHandleByUuid(_uuid, out var device) : _api.DeviceGetHandleByIndex(_index!.Value, out device);
                if (found != NvmlReturn.Success)
                {
                    if (found == NvmlReturn.Uninitialized)
                    {
                        _initialised = false;
                    }

                    return Unreadable(ClassifyDevice(found), _uuid is not null ? "nvmlDeviceGetHandleByUUID" : "nvmlDeviceGetHandleByIndex", found);
                }

                _device = device;
            }

            var call = "nvmlDeviceGetMemoryInfo_v2";
            ulong total = 0, free = 0;
            var status = NvmlReturn.FunctionNotFound;
            if (!_useV1)
            {
                status = _api.DeviceGetMemoryInfoV2(_device, out total, out free);
                _useV1 = status is NvmlReturn.FunctionNotFound or NvmlReturn.NotSupported or NvmlReturn.ArgumentVersionMismatch;
            }

            if (_useV1)
            {
                call = "nvmlDeviceGetMemoryInfo";
                status = _api.DeviceGetMemoryInfo(_device, out total, out free);
            }

            if (status == NvmlReturn.Success)
            {
                if (free > total)
                {
                    return DeviceMemoryReading.Unreadable(DeviceMemoryReadStatus.Failed, string.Create(CultureInfo.InvariantCulture, $"{call} reported {free} free of {total} bytes."));
                }

                return DeviceMemoryReading.Of(Clamp(total), Clamp(free));
            }

            var kind = ClassifyDevice(status);
            if (kind is DeviceMemoryReadStatus.DeviceLost or DeviceMemoryReadStatus.DeviceNotFound)
            {
                _device = 0; // look the device up again next time
            }

            if (status == NvmlReturn.Uninitialized)
            {
                _initialised = false;
                _device = 0;
            }

            return Unreadable(kind, call, status);
        }
    }

    /// <summary>Shuts NVML down if this reader initialised it (NVML counts initialisations).</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            if (_initialised)
            {
                _ = _api.Shutdown();
                _initialised = false;
            }
        }
    }

    /// <summary>The reading status of a failed <c>nvmlInit</c>.</summary>
    /// <param name="status">The <c>nvmlReturn_t</c>.</param>
    /// <returns>The typed status.</returns>
    public static DeviceMemoryReadStatus ClassifyInit(int status) => status switch
    {
        NvmlReturn.LibraryNotFound or NvmlReturn.FunctionNotFound or NvmlReturn.DriverNotLoaded or NvmlReturn.LibRmVersionMismatch or NvmlReturn.NoPermission
            => DeviceMemoryReadStatus.LibraryMissing,
        NvmlReturn.GpuIsLost or NvmlReturn.ResetRequired => DeviceMemoryReadStatus.DeviceLost,
        _ => DeviceMemoryReadStatus.Failed,
    };

    /// <summary>The reading status of a failed device lookup or memory read.</summary>
    /// <param name="status">The <c>nvmlReturn_t</c>.</param>
    /// <returns>The typed status.</returns>
    public static DeviceMemoryReadStatus ClassifyDevice(int status) => status switch
    {
        NvmlReturn.GpuIsLost or NvmlReturn.ResetRequired => DeviceMemoryReadStatus.DeviceLost,
        NvmlReturn.NotFound or NvmlReturn.InvalidArgument => DeviceMemoryReadStatus.DeviceNotFound,
        NvmlReturn.LibraryNotFound or NvmlReturn.DriverNotLoaded => DeviceMemoryReadStatus.LibraryMissing,
        _ => DeviceMemoryReadStatus.Failed,
    };

    private DeviceMemoryReading Unreadable(DeviceMemoryReadStatus kind, string call, int status) =>
        DeviceMemoryReading.Unreadable(kind, call + " returned " + _api.Describe(status));

    private static long Clamp(ulong bytes) => bytes > long.MaxValue ? long.MaxValue : (long)bytes;
}
