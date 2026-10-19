namespace Starlings.Roost.Cuda;

/// <summary>A CUDA runtime, driver, cuBLASLt or NVML call returned a failure status.</summary>
public sealed class CudaException : Exception
{
    /// <summary>Creates the exception.</summary>
    /// <param name="call">The function that failed.</param>
    /// <param name="status">Its status code.</param>
    /// <param name="detail">The library's name for the status, or null.</param>
    public CudaException(string call, int status, string? detail = null)
        : base($"{call} failed with status {status}{(detail is null ? string.Empty : $" ({detail})")}")
    {
        Call = call;
        Status = status;
    }

    /// <summary>The function that failed.</summary>
    public string Call { get; }

    /// <summary>The status code.</summary>
    public int Status { get; }
}

/// <summary>Status checks over the generated bindings.</summary>
public static unsafe class CudaCheck
{
    /// <summary>Throws when a CUDA runtime call did not return <c>cudaSuccess</c>.</summary>
    /// <param name="status">The call's result.</param>
    /// <param name="call">The function name, for the message.</param>
    /// <exception cref="CudaException">The status is not success.</exception>
    public static void Runtime(int status, string call)
    {
        if (status != CudaRuntime.cudaSuccess)
        {
            var name = CudaRuntime.cudaGetErrorName(status);
            throw new CudaException(call, status, name is null ? null : System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)name));
        }
    }

    /// <summary>Throws when an NVML call did not return <c>NVML_SUCCESS</c>.</summary>
    /// <param name="status">The call's result.</param>
    /// <param name="call">The function name, for the message.</param>
    /// <exception cref="CudaException">The status is not success.</exception>
    public static void Nvml(int status, string call)
    {
        if (status != Starlings.Roost.Cuda.Nvml.NVML_SUCCESS)
        {
            var name = Starlings.Roost.Cuda.Nvml.nvmlErrorString(status);
            throw new CudaException(call, status, name is null ? null : System.Runtime.InteropServices.Marshal.PtrToStringUTF8((nint)name));
        }
    }

    /// <summary>The compute capability of a device as <c>major * 10 + minor</c>, the SM number of the kernel manifest.</summary>
    /// <param name="device">CUDA device ordinal.</param>
    /// <returns>For example 86 for an RTX 3090.</returns>
    public static int ComputeCapability(int device)
    {
        int major, minor;
        Runtime(CudaRuntime.cudaDeviceGetAttribute(&major, CudaRuntime.cudaDevAttrComputeCapabilityMajor, device), "cudaDeviceGetAttribute(major)");
        Runtime(CudaRuntime.cudaDeviceGetAttribute(&minor, CudaRuntime.cudaDevAttrComputeCapabilityMinor, device), "cudaDeviceGetAttribute(minor)");
        return (major * 10) + minor;
    }
}
