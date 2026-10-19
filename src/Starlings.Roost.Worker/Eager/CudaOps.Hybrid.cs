using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Worker.Eager;

public readonly unsafe partial struct CudaOps
{
    /// <inheritdoc/>
    public static void GatedDeltaRule(in GatedDeltaRuleArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new GatedDeltaRuleParams
        {
            StructSize = (uint)sizeof(GatedDeltaRuleParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Q = a.Q,
            K = a.K,
            V = a.V,
            G = a.G,
            Beta = a.Beta,
            Output = a.Output,
            State = a.State,
            StateSlots = a.StateSlots,
            CuSeqlens = a.CuSeqlens,
            CuSeqlensHost = s.HostAddressOf(a.CuSeqlens),
            Batch = (uint)a.Batch,
            Tokens = (uint)a.Tokens,
            KHeads = (uint)a.KHeads,
            VHeads = (uint)a.VHeads,
            KDim = (uint)a.KDim,
            VDim = (uint)a.VDim,
            QStride = (uint)a.QStride,
            KStride = (uint)a.KStride,
            VStride = (uint)a.VStride,
            UseInitialState = a.UseInitialState ? 1u : 0u,
            L2NormQk = a.L2NormQk ? 1u : 0u,
            Scale = a.Scale,
        };
        s.Launch(s.Kernel(OpId.GatedDeltaRule, a.DType, preferred: RecurrentVariant), ref p);
    }

    /// <inheritdoc/>
    public static void CausalConv1d(in CausalConv1dArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new CausalConv1dParams
        {
            StructSize = (uint)sizeof(CausalConv1dParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Input = a.Input,
            Weight = a.Weight,
            Bias = a.Bias,
            Output = a.Output,
            ConvState = a.ConvState,
            StateSlots = a.StateSlots,
            CuSeqlens = a.CuSeqlens,
            Batch = (uint)a.Batch,
            Tokens = (uint)a.Tokens,
            Channels = (uint)a.Channels,
            Width = (uint)a.Width,
            Silu = a.Silu ? 1u : 0u,
        };
        s.Launch(s.Kernel(OpId.CausalConv1d, a.DType), ref p);
    }

    /// <inheritdoc/>
    public static void RecurrentStateCopy(in RecurrentStateCopyArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new RecurrentStateCopyParams
        {
            StructSize = (uint)sizeof(RecurrentStateCopyParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Pool = a.Pool,
            Pairs = a.Pairs,
            SlotBytes = (ulong)a.SlotBytes,
            Count = (uint)a.Count,
        };
        s.Launch(s.Kernel(OpId.RecurrentStateCopy, DType.Undefined), ref p);
    }

    /// <inheritdoc/>
    public static void RecurrentStateSnapshot(in RecurrentStateSnapshotArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new RecurrentStateSnapshotParams
        {
            StructSize = (uint)sizeof(RecurrentStateSnapshotParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Pool = a.Pool,
            Snapshots = a.Snapshots,
            Pairs = a.Pairs,
            SlotBytes = (ulong)a.SlotBytes,
            Count = (uint)a.Count,
            Restore = a.Restore ? 1u : 0u,
        };
        s.Launch(s.Kernel(OpId.RecurrentStateSnapshot, DType.Undefined), ref p);
    }

    /// <inheritdoc/>
    public static void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new GatedDeltaRuleGatesParams
        {
            StructSize = (uint)sizeof(GatedDeltaRuleGatesParams),
            AbiMinor = (uint)KernelAbi.Minor,
            A = a.A,
            B = a.B,
            ALog = a.ALog,
            DtBias = a.DtBias,
            G = a.G,
            Beta = a.Beta,
            Tokens = (uint)a.Tokens,
            VHeads = (uint)a.VHeads,
            AStride = (uint)a.AStride,
            BStride = (uint)a.BStride,
        };
        s.Launch(s.Kernel(OpId.GatedDeltaRuleGates, a.DType), ref p);
    }

    /// <inheritdoc/>
    public static void GatedRmsNorm(in GatedRmsNormArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new GatedRmsNormParams
        {
            StructSize = (uint)sizeof(GatedRmsNormParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Input = a.Input,
            Gate = a.Gate,
            Weight = a.Weight,
            Output = a.Output,
            Rows = (uint)a.Rows,
            Dim = (uint)a.Dim,
            Eps = a.Eps,
        };
        s.Launch(s.Kernel(OpId.GatedRmsNorm, a.DType), ref p);
    }

    /// <inheritdoc/>
    public static void SigmoidMul(in SigmoidMulArgs a)
    {
        var s = CudaOpsSession.Current;
        var p = new SigmoidMulParams
        {
            StructSize = (uint)sizeof(SigmoidMulParams),
            AbiMinor = (uint)KernelAbi.Minor,
            Input = a.Input,
            Gate = a.Gate,
            Output = a.Output,
            Rows = (uint)a.Rows,
            Dim = (uint)a.Dim,
        };
        s.Launch(s.Kernel(OpId.SigmoidMul, a.DType), ref p);
    }
}
