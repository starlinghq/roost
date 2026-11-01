using Starlings.Roost.Abi;

namespace Starlings.Roost.Conformance;

/// <summary>
/// A combination a registered variant refuses on purpose (R-019): a typed <c>UNSUPPORTED</c> from <c>plan()</c> before any launch,
/// because serving it would leave the variant's tolerance class. A rule is data of the suite, named and documented; a refusal
/// that no rule predicts is still a failure ("the variant refused a case inside its constraints").
/// </summary>
/// <param name="Variant">Variant id the rule applies to.</param>
/// <param name="Applies">Whether a case is in the refused combination.</param>
/// <param name="Status">The status the variant must return.</param>
/// <param name="Reason">Why the combination is refused, recorded in the results.</param>
public sealed record RefusalRule(string Variant, Func<OpCase, bool> Applies, KernelStatus Status, string Reason);

/// <summary>The refusals the registered variants make on purpose.</summary>
public static class RefusalRules
{
    /// <summary>
    /// <c>gemm-cublaslt</c> with a C input (<c>beta != 0</c>) or a bias, in any dtype. The maintainer's R-019 ruling of 2026-10-20 is to
    /// refuse unused GEMM epilogues at plan time; no generated forward passes C, Beta or Bias to a Gemm node
    /// (<c>RefusalTests.NoGeneratedModelForwardUsesGemmWithBetaOrBias</c> scans the model definitions), so every such epilogue is
    /// refused and alpha alone stays served. The first SM 86 run had found bf16 with C and bias outside the <c>gemm-bf16</c> class
    /// (utilisation 1.09 to 1.95, 44 elements in 5 cases). Serving an epilogue again needs a model that uses it, a kernel change and
    /// this rule removed together (kernels/README.md).
    /// </summary>
    public static readonly RefusalRule GemmEpilogueWithCOrBias = new(
        "gemm-cublaslt",
        c => c.Op == OpId.Gemm && (c.IOr("use_c", 0) != 0 || c.IOr("use_bias", 0) != 0),
        KernelStatus.Unsupported,
        "Gemm with a C input (beta != 0) or a bias: an epilogue no model uses, refused at plan time (R-019 ruling)");

    /// <summary>
    /// <c>paged-attention-flashinfer-fa2</c> with a sliding window (<c>window_left != -1</c>). The M0 kernel is built without one and
    /// returns <c>UNSUPPORTED</c> before any launch (<c>kernels/src/paged_attention.cu</c>, <c>Validate</c>, reached from the workspace
    /// query and <c>plan()</c>), as the C3 documentation of <c>PagedAttentionArgs.WindowLeft</c> records for M0. The sliding-window
    /// cases (R-113) are therefore <c>refused</c> on the GPU, never passed: the CUDA sliding-window variant and its GPU conformance are
    /// open. Remove this rule in the change that builds the window into a registered variant, so its cases are compared again.
    /// </summary>
    public static readonly RefusalRule PagedAttentionSlidingWindow = new(
        "paged-attention-flashinfer-fa2",
        c => c.Op == OpId.PagedAttention && c.IOr("window_left", -1) != -1,
        KernelStatus.Unsupported,
        "PagedAttention with a sliding window: not built in the M0 FA2 variant (R-113 CUDA variant open); the window is untested on the GPU");

    /// <summary>The rules applied by default.</summary>
    public static readonly IReadOnlyList<RefusalRule> Registered = [GemmEpilogueWithCOrBias, PagedAttentionSlidingWindow];
}
