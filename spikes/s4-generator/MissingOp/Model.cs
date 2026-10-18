using Spike.Ops;

namespace Spike.MissingOp;

// Expected to FAIL to build with ROOST001: "FlashDecode" is not a member of IRoostOps.
[RoostModel(Hidden = 16, Intermediate = 32, Vocab = 64)]
[Node("h0", "Embedding", "tokens", Weight = "embed", Out = "H")]
[Node("a0", "FlashDecode", "h0", Weight = "l0.qkv")]
[Node("logits", "LmHead", "a0", Weight = "lm_head", Out = "V")]
public static partial class BrokenModel
{
}
