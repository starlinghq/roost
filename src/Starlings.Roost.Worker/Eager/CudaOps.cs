using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Worker.Eager;

/// <summary>
/// The CUDA backend of the op vocabulary (C3, R-025): each op fills its C1 params struct from the C3 arguments and runs
/// the registered kernel on the bound session's compute stream (<see cref="CudaOpsSession"/>). It adds no arithmetic of
/// its own; what it decides is which variant runs, the shape bucket and the plan scope.
/// </summary>
/// <remarks>
/// <para>
/// What it does beyond renaming fields:
/// </para>
/// <list type="bullet">
/// <item><description>Gemm and LmHead pass <c>ShapeBucket</c> = the next power of two of M (or Selected) unless the arguments carry one, so cuBLASLt plans once per bucket and a row's result does not depend on the other rows (contract version 3).</description></item>
/// <item><description>PagedAttention sets the mode from the batch (one token per sequence is decode, more is prefill; the generated forward leaves <c>Mode</c> at its default) and reads the batch tables for <c>plan()</c> from the host mirror of the step's plan region.</description></item>
/// <item><description>LmHead writes float32 logits (C1 and <c>CpuOps</c>), whatever dtype the caller's logits span names; the span's length is taken as Selected * VocabPadded elements.</description></item>
/// <item><description>GatedDeltaRule runs the recurrent variant for every length (no plan, any token count); the chunked prefill variant is the follow-up that needs a per-step plan.</description></item>
/// <item><description>Argmax has no kernel in the library: it synchronises the compute stream, copies the logits to the host and runs <see cref="CpuOps.Argmax"/>. The fused GPU sampler is R-042.</description></item>
/// <item><description>Sample and GeluAndMul have no kernel yet and report <see cref="Supports"/> false.</description></item>
/// </list>
/// </remarks>
public readonly unsafe partial struct CudaOps : IRoostOps
{
    private const string RecurrentVariant = "gated-delta-rule-recurrent-aot";

    /// <inheritdoc/>
    public static bool Supports(OpId op) => op switch
    {
        OpId.Sample or OpId.GeluAndMul => false,
        OpId.GatedDeltaRule or OpId.CausalConv1d or OpId.RecurrentStateCopy or OpId.RecurrentStateSnapshot
            or OpId.GatedDeltaRuleGates or OpId.GatedRmsNorm or OpId.SigmoidMul => true,
        _ => OpCatalog.IsM0(op),
    };

    /// <inheritdoc/>
    public static void RmsNorm(in RmsNormArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new RmsNormParams
        {
            StructSize = (uint)sizeof(RmsNormParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Input = a.Input,
            Weight = a.Weight,
            Output = a.Output,
            Rows = (uint)a.Rows,
            Dim = (uint)a.Dim,
            Eps = a.Eps,
        };
        s.Launch(s.Kernel(OpId.RmsNorm, a.DType), ref p);
    }

    /// <inheritdoc/>
    public static void QkNorm(in QkNormArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new QkNormParams
        {
            StructSize = (uint)sizeof(QkNormParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Q = a.Q,
            K = a.K,
            QWeight = a.QWeight,
            KWeight = a.KWeight,
            Tokens = (uint)a.Tokens,
            QHeads = (uint)a.QHeads,
            KvHeads = (uint)a.KvHeads,
            HeadDim = (uint)a.HeadDim,
            Eps = a.Eps,
        };
        s.Launch(s.Kernel(OpId.QkNorm, a.DType, headDim: a.HeadDim), ref p);
    }

    /// <inheritdoc/>
    public static void Rope(in RopeArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new RopeParams
        {
            StructSize = (uint)sizeof(RopeParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Q = a.Q,
            K = a.K,
            Positions = a.Positions,
            Tokens = (uint)a.Tokens,
            QHeads = (uint)a.QHeads,
            KvHeads = (uint)a.KvHeads,
            HeadDim = (uint)a.HeadDim,
            RotaryDim = (uint)a.RotaryDim,
            Kind = a.Kind,
            Interleaved = a.Interleaved ? 1u : 0u,
            Theta = a.Theta,
            Llama3Factor = a.Llama3Factor,
            Llama3LowFreqFactor = a.Llama3LowFreqFactor,
            Llama3HighFreqFactor = a.Llama3HighFreqFactor,
            Llama3OriginalMaxPosition = (uint)a.Llama3OriginalMaxPosition,
        };
        s.Launch(s.Kernel(OpId.Rope, a.DType, headDim: a.HeadDim), ref p);
    }

    /// <inheritdoc/>
    public static void PagedAttention(in PagedAttentionArgs a)
    {
        var s = CudaOpsSession.Current;
        var mode = a.Tokens > a.Batch ? AttentionMode.Prefill : AttentionMode.Decode;
        var p = new PagedAttentionParams
        {
            StructSize = (uint)sizeof(PagedAttentionParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Q = a.Q,
            Output = a.Output,
            KCache = a.KCache,
            VCache = a.VCache,
            QIndptr = a.QIndptr,
            KvIndptr = a.KvIndptr,
            KvIndices = a.KvIndices,
            KvLastPageLen = a.KvLastPageLength,
            QIndptrHost = s.HostAddressOf(a.QIndptr),
            KvIndptrHost = s.HostAddressOf(a.KvIndptr),
            KvLastPageLenHost = s.HostAddressOf(a.KvLastPageLength),
            Mode = mode,
            Batch = (uint)a.Batch,
            Tokens = (uint)a.Tokens,
            QHeads = (uint)a.QHeads,
            KvHeads = (uint)a.KvHeads,
            HeadDim = (uint)a.HeadDim,
            PageSize = (uint)a.BlockSize,
            KvLayout = a.KvLayout,
            SmScale = a.Scale,
            LogitsSoftCap = a.SoftCap,
            WindowLeft = a.WindowLeft,
            Causal = a.Causal ? 1u : 0u,
        };
        long key = HashCode.Combine((int)mode, a.Batch, a.Tokens, a.QHeads, a.KvHeads, a.HeadDim, a.BlockSize);
        s.Launch(
            s.Kernel(OpId.PagedAttention, a.DType, kv: a.KvDType, headDim: a.HeadDim, pageSize: a.BlockSize, group: a.QHeads / a.KvHeads),
            ref p,
            PlanScope.PerStep,
            key);
    }

    /// <inheritdoc/>
    public static void SiluAndMul(in SiluAndMulArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new ActAndMulParams
        {
            StructSize = (uint)sizeof(ActAndMulParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Input = a.Input,
            Output = a.Output,
            Rows = (uint)a.Rows,
            Dim = (uint)a.Dim,
        };
        s.Launch(s.Kernel(OpId.SiluAndMul, a.DType), ref p);
    }

    /// <inheritdoc/>
    /// <remarks>No kernel is registered yet; <see cref="Supports"/> reports false.</remarks>
    public static void GeluAndMul(in GeluAndMulArgs a) => throw new NotSupportedException(nameof(GeluAndMul));

    /// <inheritdoc/>
    public static void Gemm(in GemmArgs a)
    {
        var s = CudaOpsSession.Current;
        uint bucket = a.ShapeBucket > 0 ? (uint)a.ShapeBucket : NextPowerOfTwo(a.M);
        var p = new GemmParams
        {
            StructSize = (uint)sizeof(GemmParams),
            AbiMinor = (uint)KernelAbi.Minor,
            A = a.A,
            B = a.B,
            C = a.C,
            D = a.D,
            Bias = a.Bias,
            M = (ulong)a.M,
            N = (ulong)a.N,
            K = (ulong)a.K,
            TransposeB = a.TransposeB ? 1u : 0u,
            Alpha = a.Alpha,
            Beta = a.Beta,
            ShapeBucket = bucket,
        };
        long key = HashCode.Combine(a.N, a.K, bucket, a.TransposeB, (int)a.DType, (int)a.WeightDType, a.Bias.Address != 0);
        s.Launch(s.Kernel(OpId.Gemm, a.DType, weight: a.WeightDType), ref p, PlanScope.PerShape, key);
    }

    /// <inheritdoc/>
    public static void Embedding(in EmbeddingArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new EmbeddingParams
        {
            StructSize = (uint)sizeof(EmbeddingParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Ids = a.Ids,
            Table = a.Table,
            Output = a.Output,
            Tokens = (uint)a.Tokens,
            Dim = (uint)a.Dim,
            Vocab = (uint)a.Vocab,
        };
        s.Launch(s.Kernel(OpId.Embedding, a.DType), ref p);
    }

    /// <inheritdoc/>
    public static void LmHead(in LmHeadArgs a)
    {
        var s = CudaOpsSession.Current;
        uint bucket = NextPowerOfTwo(a.Selected);
        var p = new LmHeadParams
        {
            StructSize = (uint)sizeof(LmHeadParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Hidden = a.Hidden,
            Rows = a.Rows,
            Weight = a.Weight,
            Logits = new DeviceSpan { Address = a.Logits.Address, Length = (ulong)a.Selected * (ulong)a.VocabPadded, Dtype = DType.F32 },
            Selected = (uint)a.Selected,
            Dim = (uint)a.Dim,
            Vocab = (uint)a.Vocab,
            VocabPadded = (uint)a.VocabPadded,
            SoftCap = a.SoftCap,
            ShapeBucket = bucket,
        };
        long key = HashCode.Combine(a.Dim, a.Vocab, a.VocabPadded, bucket, (int)a.DType);
        s.Launch(s.Kernel(OpId.LmHead, a.DType), ref p, PlanScope.PerShape, key);
    }

    /// <inheritdoc/>
    public static void Residual(in ResidualArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new ResidualParams
        {
            StructSize = (uint)sizeof(ResidualParams),
            AbiMinor = (uint)KernelAbi.Minor,
            A = a.A,
            B = a.B,
            Output = a.Output,
            Elements = (ulong)a.Elements,
        };
        s.Launch(s.Kernel(OpId.Residual, a.DType), ref p);
    }

    /// <inheritdoc/>
    public static void CacheWrite(in CacheWriteArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new CacheWriteParams
        {
            StructSize = (uint)sizeof(CacheWriteParams),
            AbiMinor = (uint)KernelAbi.Minor,
            K = a.K,
            V = a.V,
            KCache = a.KCache,
            VCache = a.VCache,
            SlotMapping = a.SlotMapping,
            Tokens = (uint)a.Tokens,
            KvHeads = (uint)a.KvHeads,
            HeadDim = (uint)a.HeadDim,
            PageSize = (uint)a.BlockSize,
            KvLayout = a.KvLayout,
            KScale = a.KScale == 0f ? 1f : a.KScale,
            VScale = a.VScale == 0f ? 1f : a.VScale,
        };
        s.Launch(s.Kernel(OpId.CacheWrite, a.KvDType, headDim: a.HeadDim, pageSize: a.BlockSize), ref p);
    }

    /// <inheritdoc/>
    public static void CacheCopy(in CacheCopyArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new CacheCopyParams
        {
            StructSize = (uint)sizeof(CacheCopyParams),
            AbiMinor = (uint)KernelAbi.Minor,
            SrcPool = a.Source,
            DstPool = a.Destination,
            Pairs = a.Pairs,
            BlockBytes = (ulong)a.BlockBytes,
            Count = (uint)a.Count,
        };
        s.Launch(s.Kernel(OpId.CacheCopy, DType.Undefined), ref p);
    }

    /// <inheritdoc/>
    /// <remarks>Host fallback: see the type remarks.</remarks>
    public static void Argmax(in ArgmaxArgs a)
    {
        var s = CudaOpsSession.Current;
        int stride = a.Stride == 0 ? a.Vocab : a.Stride;
        long elements = ((long)(a.Rows - 1) * stride) + a.Vocab;
        long logitBytes = elements * ElementBytes(a.Logits.Dtype);
        long idBytes = (long)a.Rows * ElementBytes(a.Ids.Dtype);
        void* logits = NativeMemory.Alloc((nuint)logitBytes);
        void* ids = NativeMemory.Alloc((nuint)idBytes);
        try
        {
            CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(s.Stream), "cudaStreamSynchronize");
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy(logits, (void*)a.Logits.Address, (nuint)logitBytes, CudaRuntime.cudaMemcpyDeviceToHost), "cudaMemcpy(logits)");
            var host = a;
            host.Logits = new DeviceSpan { Address = (nint)logits, Length = (ulong)elements, Dtype = a.Logits.Dtype };
            host.Ids = new DeviceSpan { Address = (nint)ids, Length = (ulong)a.Rows, Dtype = a.Ids.Dtype };
            CpuOps.Argmax(in host);
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy((void*)a.Ids.Address, ids, (nuint)idBytes, CudaRuntime.cudaMemcpyHostToDevice), "cudaMemcpy(ids)");
        }
        finally
        {
            NativeMemory.Free(logits);
            NativeMemory.Free(ids);
        }
    }

    /// <inheritdoc/>
    /// <remarks>No kernel is registered yet (the fused sampler is R-042); <see cref="Supports"/> reports false.</remarks>
    public static void Sample(in SampleArgs a) => throw new NotSupportedException(nameof(Sample));

    /// <summary>The smallest power of two that is at least <paramref name="n"/> (1 for n &lt; 1).</summary>
    /// <param name="n">A row count.</param>
    /// <returns>The bucket.</returns>
    public static uint NextPowerOfTwo(long n) => n <= 1 ? 1u : (uint)System.Numerics.BitOperations.RoundUpToPowerOf2((ulong)n);

    private static long ElementBytes(DType dtype) => dtype switch
    {
        DType.F32 or DType.I32 => 4,
        DType.F16 or DType.BF16 => 2,
        DType.I64 => 8,
        _ => 1,
    };
}
