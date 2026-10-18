using Spike.Ops;

namespace Spike.Toy;

// Two decoder-style layers (RmsNorm, Gemm, SiluAndMul, Gemm, Residual) and an LmHead. Attention is omitted: the spike
// is about the generator, the graph and the dispatch, not about the op set.
[RoostModel(Hidden = 16, Intermediate = 32, Vocab = 64)]
[Node("h0", "Embedding", "tokens", Weight = "embed", Out = "H")]
[Node("l0_norm1", "RmsNorm", "h0", Weight = "l0.norm1")]
[Node("l0_up", "Gemm", "l0_norm1", Weight = "l0.gate_up", Out = "2I")]
[Node("l0_act", "SiluAndMul", "l0_up")]
[Node("l0_down", "Gemm", "l0_act", Weight = "l0.down", Out = "H")]
[Node("h1", "Residual", "h0", "l0_down")]
[Node("l1_norm1", "RmsNorm", "h1", Weight = "l1.norm1")]
[Node("l1_up", "Gemm", "l1_norm1", Weight = "l1.gate_up", Out = "2I")]
[Node("l1_act", "SiluAndMul", "l1_up")]
[Node("l1_down", "Gemm", "l1_act", Weight = "l1.down", Out = "H")]
[Node("h2", "Residual", "h1", "l1_down")]
[Node("final_norm", "RmsNorm", "h2", Weight = "final.norm")]
[Node("logits", "LmHead", "final_norm", Weight = "lm_head", Out = "V")]
public static partial class ToyModel
{
}
