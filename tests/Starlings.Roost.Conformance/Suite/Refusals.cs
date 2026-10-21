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
    /// <c>gemm-cublaslt</c> in bf16 with a C input (<c>beta != 0</c>) and a bias together. On SM 86 the pinned algorithms exceed the
    /// <c>gemm-bf16</c> class on it (utilisation 1.09 to 1.95, 44 elements in 5 conformance cases); f16 and every other epilogue
    /// pass. No generated forward uses the combination (<c>ModelForwardsTests</c>), so the variant refuses it instead of the class
    /// being widened. The kernel fix is a follow-up (kernels/README.md).
    /// </summary>
    public static readonly RefusalRule GemmBf16CWithBias = new(
        "gemm-cublaslt",
        c => c.Op == OpId.Gemm && c.Dtype == DType.BF16 && c.IOr("use_c", 0) != 0 && c.IOr("use_bias", 0) != 0,
        KernelStatus.Unsupported,
        "bf16 Gemm with beta != 0 and a bias is outside the gemm-bf16 class; refused at plan time (R-019 finding)");

    /// <summary>The rules applied by default.</summary>
    public static readonly IReadOnlyList<RefusalRule> Registered = [GemmBf16CWithBias];
}
