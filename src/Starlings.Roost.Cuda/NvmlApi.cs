namespace Starlings.Roost.Cuda;

/// <summary>
/// The <c>nvmlReturn_t</c> codes the NVML memory reader tells apart (R-044) beyond the ones the generator emits into
/// <see cref="Nvml"/>. Hand-written: their values are <c>nvml.h</c>'s, and they move into
/// <c>tools/generate-cuda-bindings.py</c>'s constant list (stream A), where they are checked against the pinned header, the
/// next time the bindings are regenerated.
/// </summary>
public static class NvmlReturn
{
    /// <summary><c>NVML_SUCCESS</c>.</summary>
    public const int Success = Nvml.NVML_SUCCESS;

    /// <summary><c>NVML_ERROR_UNINITIALIZED</c>: <c>nvmlInit</c> has not run, or was undone.</summary>
    public const int Uninitialized = Nvml.NVML_ERROR_UNINITIALIZED;

    /// <summary><c>NVML_ERROR_INVALID_ARGUMENT</c>.</summary>
    public const int InvalidArgument = 2;

    /// <summary><c>NVML_ERROR_NOT_SUPPORTED</c>.</summary>
    public const int NotSupported = Nvml.NVML_ERROR_NOT_SUPPORTED;

    /// <summary><c>NVML_ERROR_NO_PERMISSION</c>.</summary>
    public const int NoPermission = Nvml.NVML_ERROR_NO_PERMISSION;

    /// <summary><c>NVML_ERROR_NOT_FOUND</c>: no device with that index or UUID.</summary>
    public const int NotFound = 6;

    /// <summary><c>NVML_ERROR_DRIVER_NOT_LOADED</c>.</summary>
    public const int DriverNotLoaded = 9;

    /// <summary><c>NVML_ERROR_LIBRARY_NOT_FOUND</c>; <see cref="NvmlApi"/> also returns it when the library cannot be loaded.</summary>
    public const int LibraryNotFound = 12;

    /// <summary><c>NVML_ERROR_FUNCTION_NOT_FOUND</c>; <see cref="NvmlApi"/> also returns it when an entry point is missing.</summary>
    public const int FunctionNotFound = 13;

    /// <summary><c>NVML_ERROR_GPU_IS_LOST</c>: the device fell off the bus.</summary>
    public const int GpuIsLost = 15;

    /// <summary><c>NVML_ERROR_RESET_REQUIRED</c>.</summary>
    public const int ResetRequired = 16;

    /// <summary><c>NVML_ERROR_LIB_RM_VERSION_MISMATCH</c>: the library and the kernel driver disagree.</summary>
    public const int LibRmVersionMismatch = Nvml.NVML_ERROR_LIB_RM_VERSION_MISMATCH;

    /// <summary><c>NVML_ERROR_ARGUMENT_VERSION_MISMATCH</c>: a versioned struct the driver does not know.</summary>
    public const int ArgumentVersionMismatch = 25;

    /// <summary><c>NVML_ERROR_UNKNOWN</c>.</summary>
    public const int Unknown = 999;
}

/// <summary>
/// The NVML calls the device memory reader makes (R-044), behind an interface so the reader's logic is tested on the CPU
/// with a fake. Every member returns an <c>nvmlReturn_t</c> and never throws.
/// </summary>
public interface INvmlApi
{
    /// <summary><c>nvmlInit_v2</c>.</summary>
    /// <returns>The status.</returns>
    int Init();

    /// <summary><c>nvmlShutdown</c>.</summary>
    /// <returns>The status.</returns>
    int Shutdown();

    /// <summary><c>nvmlDeviceGetHandleByIndex_v2</c>. NVML indexes devices in PCI bus order, which matches CUDA ordinals only under <c>CUDA_DEVICE_ORDER=PCI_BUS_ID</c>.</summary>
    /// <param name="index">The NVML index.</param>
    /// <param name="device">The handle on success.</param>
    /// <returns>The status.</returns>
    int DeviceGetHandleByIndex(uint index, out nint device);

    /// <summary><c>nvmlDeviceGetHandleByUUID</c>.</summary>
    /// <param name="uuid">The device UUID, for example <c>GPU-…</c>.</param>
    /// <param name="device">The handle on success.</param>
    /// <returns>The status.</returns>
    int DeviceGetHandleByUuid(string uuid, out nint device);

    /// <summary><c>nvmlDeviceGetMemoryInfo_v2</c>, whose free bytes exclude memory the driver reserves.</summary>
    /// <param name="device">The handle.</param>
    /// <param name="totalBytes">Total bytes on success.</param>
    /// <param name="freeBytes">Free bytes on success.</param>
    /// <returns>The status.</returns>
    int DeviceGetMemoryInfoV2(nint device, out ulong totalBytes, out ulong freeBytes);

    /// <summary><c>nvmlDeviceGetMemoryInfo</c>, for drivers without the v2 call.</summary>
    /// <param name="device">The handle.</param>
    /// <param name="totalBytes">Total bytes on success.</param>
    /// <param name="freeBytes">Free bytes on success.</param>
    /// <returns>The status.</returns>
    int DeviceGetMemoryInfo(nint device, out ulong totalBytes, out ulong freeBytes);

    /// <summary><c>nvmlErrorString</c>, or a plain description when the library is missing.</summary>
    /// <param name="status">A status.</param>
    /// <returns>The library's name for it.</returns>
    string Describe(int status);
}

/// <summary>
/// <see cref="INvmlApi"/> over the generated <see cref="Nvml"/> bindings. Loading NVML creates no CUDA context. A library
/// that cannot be loaded is <see cref="NvmlReturn.LibraryNotFound"/> and a missing entry point
/// <see cref="NvmlReturn.FunctionNotFound"/>, never an exception.
/// </summary>
public sealed unsafe class NvmlApi : INvmlApi
{
    private NvmlApi()
    {
    }

    /// <summary>The process-wide instance. Installs the <see cref="CudaLibraryResolver"/> on first use.</summary>
    public static NvmlApi Instance { get; } = new();

    /// <inheritdoc />
    public int Init()
    {
        try
        {
            CudaLibraryResolver.Install();
            return Nvml.nvmlInit();
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            return StatusOf(e);
        }
    }

    /// <inheritdoc />
    public int Shutdown()
    {
        try
        {
            return Nvml.nvmlShutdown();
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            return StatusOf(e);
        }
    }

    /// <inheritdoc />
    public int DeviceGetHandleByIndex(uint index, out nint device)
    {
        nint handle = 0;
        int status;
        try
        {
            status = Nvml.nvmlDeviceGetHandleByIndex(index, &handle);
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            status = StatusOf(e);
        }

        device = handle;
        return status;
    }

    /// <inheritdoc />
    public int DeviceGetHandleByUuid(string uuid, out nint device)
    {
        ArgumentNullException.ThrowIfNull(uuid);
        nint handle = 0;
        int status;
        try
        {
            status = Nvml.nvmlDeviceGetHandleByUUID(uuid, &handle);
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            status = StatusOf(e);
        }

        device = handle;
        return status;
    }

    /// <inheritdoc />
    public int DeviceGetMemoryInfoV2(nint device, out ulong totalBytes, out ulong freeBytes)
    {
        var memory = new nvmlMemory_v2_t { version = Nvml.nvmlMemory_v2 };
        int status;
        try
        {
            status = Nvml.nvmlDeviceGetMemoryInfo_v2(device, &memory);
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            status = StatusOf(e);
        }

        totalBytes = memory.total;
        freeBytes = memory.free;
        return status;
    }

    /// <inheritdoc />
    public int DeviceGetMemoryInfo(nint device, out ulong totalBytes, out ulong freeBytes)
    {
        var memory = default(nvmlMemory_t);
        int status;
        try
        {
            status = Nvml.nvmlDeviceGetMemoryInfo(device, &memory);
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            status = StatusOf(e);
        }

        totalBytes = memory.total;
        freeBytes = memory.free;
        return status;
    }

    /// <inheritdoc />
    public string Describe(int status)
    {
        try
        {
            var name = Nvml.nvmlErrorString(status);
            return name is null ? $"status {status}" : $"status {status} ({System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)name)})";
        }
        catch (Exception e) when (IsUnavailable(e))
        {
            return $"status {status}";
        }
    }

    private static bool IsUnavailable(Exception e) => e is DllNotFoundException or BadImageFormatException or EntryPointNotFoundException;

    private static int StatusOf(Exception e) => e is EntryPointNotFoundException ? NvmlReturn.FunctionNotFound : NvmlReturn.LibraryNotFound;
}
