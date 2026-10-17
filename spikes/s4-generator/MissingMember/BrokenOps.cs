using Spike.Ops;

namespace Spike.MissingMember;

// Expected to FAIL to build with CS0535: the backend does not implement the Gemm member of IRoostOps.
public struct BrokenOps : IRoostOps
{
    public static void Embedding(in EmbeddingArgs a) { }
    public static void RmsNorm(in RmsNormArgs a) { }
    public static void SiluAndMul(in SiluAndMulArgs a) { }
    public static void Residual(in ResidualArgs a) { }
    public static void LmHead(in LmHeadArgs a) { }
}
