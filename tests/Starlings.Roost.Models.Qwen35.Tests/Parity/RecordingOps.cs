using Starlings.Roost.Abi;
using Starlings.Roost.Ops;

namespace Starlings.Roost.Models.Qwen35.Tests.Parity;

/// <summary>
/// A backend that runs every op on <see cref="CpuOps"/> and, when <see cref="StageRecorder.Active"/> is set, copies out what the
/// per-layer stage (R-011 layout) needs: the embedding output, each decoder layer's output and the final norm output, and every
/// Gated DeltaNet layer's recurrent and convolution state. The generated forward shares its arena by live range, so these values
/// are gone once Forward returns; recording them as they are produced is the only way to read them without changing the model.
/// </summary>
/// <remarks>
/// Stages are identified by call order within one forward of <see cref="Qwen35Model"/>: the one Embedding call is the embeddings;
/// Residual calls alternate the token-mixer residual and the MLP residual, so the 2nd, 4th, ... 48th are layer.0 .. layer.23; the
/// 49th RmsNorm (after an input and a post-attention norm in each of 24 layers) is the final norm; the n-th CausalConv1d and
/// GatedDeltaRule calls belong to the n-th linear-attention layer. <see cref="StageRecorder.EndForward"/> checks the counts.
/// A <see cref="Perturbation"/> may change one op, to show what a real defect looks like against the proposed bounds; the model
/// definition is never changed.
/// </remarks>
internal readonly struct RecordingOps : IRoostOps
{
    public static bool Supports(OpId op) => CpuOps.Supports(op);

    public static void RmsNorm(in RmsNormArgs a)
    {
        if (StageRecorder.Active?.Perturbation == Perturbation.NormEps)
        {
            var b = a;
            b.Eps = Perturbations.WrongEps;
            CpuOps.RmsNorm(in b);
        }
        else
        {
            CpuOps.RmsNorm(in a);
        }

        StageRecorder.Active?.OnRmsNorm(in a);
    }

    public static void QkNorm(in QkNormArgs a)
    {
        if (StageRecorder.Active?.Perturbation == Perturbation.NormEps)
        {
            var b = a;
            b.Eps = Perturbations.WrongEps;
            CpuOps.QkNorm(in b);
            return;
        }

        CpuOps.QkNorm(in a);
    }

    public static void Rope(in RopeArgs a) => CpuOps.Rope(in a);

    public static void PagedAttention(in PagedAttentionArgs a) => CpuOps.PagedAttention(in a);

    public static void SiluAndMul(in SiluAndMulArgs a) => CpuOps.SiluAndMul(in a);

    public static void GeluAndMul(in GeluAndMulArgs a) => CpuOps.GeluAndMul(in a);

    public static void Gemm(in GemmArgs a) => CpuOps.Gemm(in a);

    public static void Embedding(in EmbeddingArgs a)
    {
        CpuOps.Embedding(in a);
        StageRecorder.Active?.OnEmbedding(in a);
    }

    public static void LmHead(in LmHeadArgs a) => CpuOps.LmHead(in a);

    public static void Residual(in ResidualArgs a)
    {
        CpuOps.Residual(in a);
        StageRecorder.Active?.OnResidual(in a);
    }

    public static void CacheWrite(in CacheWriteArgs a) => CpuOps.CacheWrite(in a);

    public static void CacheCopy(in CacheCopyArgs a) => CpuOps.CacheCopy(in a);

    public static void Argmax(in ArgmaxArgs a) => CpuOps.Argmax(in a);

    public static void Sample(in SampleArgs a) => CpuOps.Sample(in a);

    public static void GatedDeltaRule(in GatedDeltaRuleArgs a)
    {
        CpuOps.GatedDeltaRule(in a);
        StageRecorder.Active?.OnGatedDeltaRule(in a);
    }

    public static void CausalConv1d(in CausalConv1dArgs a)
    {
        // The reference keeps the last Width inputs; the slot keeps Width - 1. Read the incoming slot before the op overwrites it.
        StageRecorder.Active?.BeforeCausalConv1d(in a);
        CpuOps.CausalConv1d(in a);
    }

    public static void RecurrentStateCopy(in RecurrentStateCopyArgs a) => CpuOps.RecurrentStateCopy(in a);

    public static void RecurrentStateSnapshot(in RecurrentStateSnapshotArgs a) => CpuOps.RecurrentStateSnapshot(in a);

    public static void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a) => CpuOps.GatedDeltaRuleGates(in a);

    public static void GatedRmsNorm(in GatedRmsNormArgs a)
    {
        switch (StageRecorder.Active?.Perturbation)
        {
            case Perturbation.NormEps:
            {
                var b = a;
                b.Eps = Perturbations.WrongEps;
                CpuOps.GatedRmsNorm(in b);
                return;
            }

            case Perturbation.DropZGate:
                Perturbations.RmsNormWithoutGate(in a);
                return;
            default:
                CpuOps.GatedRmsNorm(in a);
                return;
        }
    }

    public static void SigmoidMul(in SigmoidMulArgs a)
    {
        if (StageRecorder.Active?.Perturbation == Perturbation.DropAttentionGate)
        {
            Perturbations.Copy(a.Input, a.Output, (long)a.Rows * a.Dim);
            return;
        }

        CpuOps.SigmoidMul(in a);
    }
}
