using Starlings.Roost.Abi;
using Starlings.Roost.Models.Qwen35;
using Starlings.Roost.Ops;
using Starlings.Roost.Worker.Tests.Support;

namespace Starlings.Roost.Worker.Tests;

/// <summary>A backend that does nothing and remembers what the generated forward asked of PagedAttention.</summary>
internal readonly struct RecordingOps : IRoostOps
{
    [ThreadStatic]
    private static List<int>? s_windows;

    public static List<int>? Windows
    {
        get => s_windows;
        set => s_windows = value;
    }

    public static void RmsNorm(in RmsNormArgs a)
    {
    }

    public static void QkNorm(in QkNormArgs a)
    {
    }

    public static void Rope(in RopeArgs a)
    {
    }

    public static void PagedAttention(in PagedAttentionArgs a) => s_windows?.Add(a.WindowLeft);

    public static void SiluAndMul(in SiluAndMulArgs a)
    {
    }

    public static void GeluAndMul(in GeluAndMulArgs a)
    {
    }

    public static void Gemm(in GemmArgs a)
    {
    }

    public static void Embedding(in EmbeddingArgs a)
    {
    }

    public static void LmHead(in LmHeadArgs a)
    {
    }

    public static void Residual(in ResidualArgs a)
    {
    }

    public static void CacheWrite(in CacheWriteArgs a)
    {
    }

    public static void CacheCopy(in CacheCopyArgs a)
    {
    }

    public static void Argmax(in ArgmaxArgs a)
    {
    }

    public static void Sample(in SampleArgs a)
    {
    }

    public static bool Supports(OpId op) => true;

    public static void GatedDeltaRule(in GatedDeltaRuleArgs a)
    {
    }

    public static void CausalConv1d(in CausalConv1dArgs a)
    {
    }

    public static void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a)
    {
    }

    public static void GatedRmsNorm(in GatedRmsNormArgs a)
    {
    }

    public static void SigmoidMul(in SigmoidMulArgs a)
    {
    }
}

public sealed unsafe class GeneratedForwardTests
{
    private static void RunForward<TOps>()
        where TOps : struct, IRoostOps
    {
        var layout = new Qwen35Model.ForwardLayout(8, 1, 2);
        var weights = new DeviceSpan[Qwen35Model.WeightSlots.Length];
        var state = new DeviceSpan[Qwen35Model.StateSlots.Length];
        var plan = Qwen35Model.PlanInputs.Bind(default, layout, 1, 1, 1, 1);
        var ctx = new Qwen35Model.StepContext(layout, default, weights, state, plan, 1, 1, 1, 1);
        Qwen35Model.Forward<TOps>(in ctx);
    }

    [Fact]
    public void TheGeneratedForwardBindsNoWindowForEveryAttentionLayer()
    {
        // R-025 found WindowLeft unbound (0, a one-token window); R-035 binds -1 in the model definition.
        RecordingOps.Windows = [];
        try
        {
            RunForward<RecordingOps>();
            Assert.Equal(6, RecordingOps.Windows.Count);
            Assert.All(RecordingOps.Windows, w => Assert.Equal(-1, w));
        }
        finally
        {
            RecordingOps.Windows = null;
        }
    }

    [Fact]
    public void ProbePassesTheGeneratedWindowThrough()
    {
        RecordingOps.Windows = [];
        try
        {
            RunForward<Probe<RecordingOps>>();
            Assert.Equal(6, RecordingOps.Windows.Count);
            Assert.All(RecordingOps.Windows, w => Assert.Equal(-1, w));
        }
        finally
        {
            RecordingOps.Windows = null;
        }
    }

    [Fact]
    public void AWindowOfZeroMakesCpuOpsAttendToTheCurrentKeyOnly()
    {
        // Three keys with distinct values; the last query. WindowLeft 0 gives V[2]; -1 gives the softmax mix of all three.
        const int hd = 4;
        var q = new float[hd] { 1, 0, 0, 0 };
        var k = new float[3 * hd];
        var v = new float[3 * hd];
        for (int i = 0; i < 3; i++)
        {
            k[i * hd] = i;
            v[i * hd] = 10 * (i + 1);
        }

        float[] Attend(int window)
        {
            var o = new float[hd];
            var ids = new int[] { 0 };
            var qIndptr = new int[] { 0, 1 };
            var kvIndptr = new int[] { 0, 1 };
            var last = new int[] { 3 };
            fixed (float* pq = q, pk = k, pv = v, po = o)
            fixed (int* pi = ids, pqi = qIndptr, pki = kvIndptr, pl = last)
            {
                CpuOps.PagedAttention(new PagedAttentionArgs
                {
                    Q = new DeviceSpan { Address = (nint)pq, Length = hd, Dtype = DType.F32 },
                    Output = new DeviceSpan { Address = (nint)po, Length = hd, Dtype = DType.F32 },
                    KCache = new DeviceSpan { Address = (nint)pk, Length = (ulong)k.Length, Dtype = DType.F32 },
                    VCache = new DeviceSpan { Address = (nint)pv, Length = (ulong)v.Length, Dtype = DType.F32 },
                    QIndptr = new DeviceSpan { Address = (nint)pqi, Length = 2, Dtype = DType.I32 },
                    KvIndptr = new DeviceSpan { Address = (nint)pki, Length = 2, Dtype = DType.I32 },
                    KvIndices = new DeviceSpan { Address = (nint)pi, Length = 1, Dtype = DType.I32 },
                    KvLastPageLength = new DeviceSpan { Address = (nint)pl, Length = 1, Dtype = DType.I32 },
                    Batch = 1,
                    Tokens = 1,
                    QHeads = 1,
                    KvHeads = 1,
                    HeadDim = hd,
                    BlockSize = 16,
                    KvLayout = TensorLayout.PagedNhd,
                    Scale = 1f,
                    Causal = true,
                    WindowLeft = window,
                    KvDType = DType.F32,
                    DType = DType.F32,
                });
            }

            return o;
        }

        Assert.Equal(30f, Attend(0)[0]);
        float mix = Attend(-1)[0];
        Assert.InRange(mix, 20f, 30f);
        Assert.NotEqual(30f, mix);
    }
}
