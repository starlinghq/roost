using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Qwen35.Tests.Parity;

/// <summary>Deliberate defects for the R-022 negative runs: each changes one op in <see cref="RecordingOps"/>, never the definition.</summary>
internal enum Perturbation
{
    /// <summary>The model as defined.</summary>
    None,

    /// <summary>The attention output gate is dropped: <c>SigmoidMul</c> copies its input (attention layers 3, 7, ... 23).</summary>
    DropAttentionGate,

    /// <summary>The linear layers' gated norm drops <c>silu(z)</c>: plain RMS norm times its weight (linear layers 0, 1, 2, 4, ...).</summary>
    DropZGate,

    /// <summary>Every RMS norm (decoder, q/k, gated, final) uses eps 1e-5 instead of the checkpoint's 1e-6.</summary>
    NormEps,
}

internal static unsafe class Perturbations
{
    /// <summary>The wrong eps of <see cref="Perturbation.NormEps"/> (a common default; the checkpoint's rms_norm_eps is 1e-6).</summary>
    public const float WrongEps = 1e-5f;

    public static Perturbation Parse(string? text) => text switch
    {
        null or "" or "none" => Perturbation.None,
        "drop-attention-gate" => Perturbation.DropAttentionGate,
        "drop-z-gate" => Perturbation.DropZGate,
        "norm-eps-1e-5" => Perturbation.NormEps,
        _ => throw new ArgumentException($"unknown perturbation '{text}' (none, drop-attention-gate, drop-z-gate, norm-eps-1e-5)"),
    };

    public static string Name(Perturbation p) => p switch
    {
        Perturbation.DropAttentionGate => "drop-attention-gate",
        Perturbation.DropZGate => "drop-z-gate",
        Perturbation.NormEps => "norm-eps-1e-5",
        _ => "none",
    };

    /// <summary>Copies <paramref name="elements"/> elements of <paramref name="from"/>'s type into <paramref name="to"/>.</summary>
    public static void Copy(DeviceSpan from, DeviceSpan to, long elements)
    {
        long bytes = elements * (from.Dtype == DType.F32 ? 4 : 2);
        Buffer.MemoryCopy((void*)from.Address, (void*)to.Address, bytes, bytes);
    }

    /// <summary><see cref="CpuOps.GatedRmsNorm"/> without the <c>silu(Gate)</c> factor: <c>x / sqrt(mean(x^2) + Eps) * Weight</c>.</summary>
    public static void RmsNormWithoutGate(in GatedRmsNormArgs a)
    {
        if (a.DType != DType.BF16)
        {
            throw new NotSupportedException("the perturbation is written for bfloat16 activations");
        }

        DType wt = a.Weight.Dtype == DType.Undefined ? a.DType : a.Weight.Dtype;
        for (long r = 0; r < a.Rows; r++)
        {
            long o = r * a.Dim;
            double ss = 0;
            for (int i = 0; i < a.Dim; i++)
            {
                double v = StageRecorder.Load(a.Input.Address, a.DType, o + i);
                ss += v * v;
            }

            double inv = 1.0 / Math.Sqrt((ss / a.Dim) + a.Eps);
            for (int i = 0; i < a.Dim; i++)
            {
                double y = StageRecorder.Load(a.Input.Address, a.DType, o + i) * inv * StageRecorder.Load(a.Weight.Address, wt, i);
                ((ushort*)a.Output.Address)[o + i] = ToBFloat16((float)y);
            }
        }
    }

    // Round to nearest even; finite inputs only (a perturbation, not a reference).
    private static ushort ToBFloat16(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        u += 0x7FFFu + ((u >> 16) & 1u);
        return (ushort)(u >> 16);
    }
}
