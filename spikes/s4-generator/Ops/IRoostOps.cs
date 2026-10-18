using System;

namespace Spike.Ops;

/// <summary>Element type tag carried by every buffer (C3: argument structs carry dtype).</summary>
public enum DType : byte { F32 = 0, I32 = 1 }

/// <summary>Pointer, length and dtype; there is no tensor type (C3). Blittable.</summary>
public readonly struct DeviceSpan
{
    public DeviceSpan(nint ptr, int length, DType dtype)
    {
        Ptr = ptr;
        Length = length;
        Dtype = dtype;
    }

    public nint Ptr { get; }
    public int Length { get; }
    public DType Dtype { get; }
}

public struct EmbeddingArgs { public DeviceSpan Input; public DeviceSpan Weight; public DeviceSpan Output; public int Rows; public int OutDim; }
public struct RmsNormArgs { public DeviceSpan Input; public DeviceSpan Weight; public DeviceSpan Output; public int Rows; public int InDim; }
public struct GemmArgs { public DeviceSpan Input; public DeviceSpan Weight; public DeviceSpan Output; public int Rows; public int InDim; public int OutDim; }
public struct SiluAndMulArgs { public DeviceSpan Input; public DeviceSpan Output; public int Rows; public int OutDim; }
public struct ResidualArgs { public DeviceSpan Input; public DeviceSpan Input2; public DeviceSpan Output; public int Rows; public int InDim; }
public struct LmHeadArgs { public DeviceSpan Input; public DeviceSpan Weight; public DeviceSpan Output; public int Rows; public int InDim; public int OutDim; }

/// <summary>Optional fused op (C3): Output2 = Input + Input2, Output = RmsNorm(Output2, Weight).</summary>
public struct AddRmsNormArgs { public DeviceSpan Input; public DeviceSpan Input2; public DeviceSpan Weight; public DeviceSpan Output; public DeviceSpan Output2; public int Rows; public int InDim; }

/// <summary>Spike-local subset of the C3 op vocabulary: static abstract members, one per op.</summary>
public interface IRoostOps
{
    static abstract void Embedding(in EmbeddingArgs a);
    static abstract void RmsNorm(in RmsNormArgs a);
    static abstract void Gemm(in GemmArgs a);
    static abstract void SiluAndMul(in SiluAndMulArgs a);
    static abstract void Residual(in ResidualArgs a);
    static abstract void LmHead(in LmHeadArgs a);

    /// <summary>
    /// Optional fused op. A static virtual default cannot call the other static abstract members without a self type
    /// parameter, so the fallback is chosen by the generated forward: it emits
    /// <c>if (TOps.SupportsAddRmsNorm)</c> over the fused call and the unfused parts, and the JIT and ILC fold the
    /// constant per struct instantiation.
    /// </summary>
    static virtual bool SupportsAddRmsNorm => false;

    /// <summary>Fused add and RmsNorm; called only when <see cref="SupportsAddRmsNorm"/> is true.</summary>
    static virtual void AddRmsNorm(in AddRmsNormArgs a) => throw new NotSupportedException();
}
