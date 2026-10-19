using Starlings.Roost.Abi;

namespace Starlings.Roost.Kernels;

/// <summary>
/// One entry of a loaded library's registration table (<c>RoostOpEntry</c>), copied out of native memory. The function
/// pointers stay valid until the owning <see cref="KernelLibrary"/> is disposed.
/// </summary>
/// <param name="Op">The op.</param>
/// <param name="VariantId">Stable ASCII id, unique per op across loaded libraries.</param>
/// <param name="Flags">Registration flags.</param>
/// <param name="Constraints">Where the variant is valid.</param>
/// <param name="ParamsStructSize">Size of the op's params struct the library was built against.</param>
/// <param name="ParamsAbiMinor">The minor of that struct.</param>
/// <param name="Run">Address of the <c>run</c> entry point (<c>RoostRunFn</c>).</param>
/// <param name="Plan">Address of <c>plan</c>, or 0.</param>
/// <param name="Workspace">Address of the workspace query, or 0.</param>
/// <param name="Library">The library that registered it.</param>
public sealed record RegisteredKernel(
    OpId Op,
    string VariantId,
    VariantTraits Flags,
    KernelConstraints Constraints,
    uint ParamsStructSize,
    uint ParamsAbiMinor,
    nint Run,
    nint Plan,
    nint Workspace,
    KernelLibrary Library)
{
    /// <summary>Calls <c>run</c> (<c>RoostStatus (*)(RoostContext*, const void* params, const RoostWorkspace*, void* stream)</c>).</summary>
    /// <param name="context">A context from <see cref="KernelLibrary.CreateContext"/> of the same library.</param>
    /// <param name="parameters">The op's params struct.</param>
    /// <param name="workspace">The caller-owned workspace, or null when the workspace query reports zero sizes.</param>
    /// <param name="stream">The CUDA stream (<c>cudaStream_t</c>).</param>
    /// <returns>The library's status.</returns>
    public unsafe KernelStatus RunKernel(KernelContext context, void* parameters, Workspace* workspace, nint stream)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (!ReferenceEquals(context.Library, Library))
        {
            throw new ArgumentException("the context belongs to another kernel library", nameof(context));
        }

        return (KernelStatus)((delegate* unmanaged<nint, void*, Workspace*, nint, int>)Run)(context.Handle, parameters, workspace, stream);
    }
}
