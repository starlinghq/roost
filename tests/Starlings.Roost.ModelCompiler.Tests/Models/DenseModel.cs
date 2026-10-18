using Starlings.Roost.ModelCompiler;

namespace Starlings.Roost.ModelCompiler.Tests.Models;

// A two-layer dense decoder without attention (the attention path is exercised by HybridModel): embedding, then per
// layer RmsNorm, a fused gate/up projection, SiluAndMul, a down projection and a residual add, then the final norm
// and the head. float32 so the test stub backend can run it. It is the S4 toy on the real C3 argument structs.
[RoostModel("tiny-dense", Dtype = "F32")]
[Dim("hidden", 16)]
[Dim("inter", 32)]
[Dim("inter2", "2*inter")]
[Dim("vocab", 64)]
[Weight("embed", "vocab,hidden", Source = "model.embed_tokens.weight")]
[Weight("l{i}.norm", "hidden", Source = "model.layers.{i}.input_layernorm.weight", Layers = "0:2")]
[Weight("l{i}.gate_up", "inter2,hidden", Source = "model.layers.{i}.mlp.gate_up_proj.weight", Layers = "0:2")]
[Weight("l{i}.down", "hidden,inter", Source = "model.layers.{i}.mlp.down_proj.weight", Layers = "0:2")]
[Weight("final_norm", "hidden", Source = "model.norm.weight")]
[Weight("lm_head", "vocab,hidden", Source = "lm_head.weight")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
[Node("l{i}.norm", "RmsNorm", "Input=h{i}; Weight=l{i}.norm; Output=>l{i}.normed[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:2")]
[Node("l{i}.up", "Gemm", "A=l{i}.normed; B=l{i}.gate_up; D=>l{i}.up[tokens,inter2]; M=tokens; N=inter2; K=hidden; TransposeB=true; Alpha=1; Beta=0; WeightDType=F32", Layers = "0:2")]
[Node("l{i}.act", "SiluAndMul", "Input=l{i}.up; Output=>l{i}.act[tokens,inter]; Rows=tokens; Dim=inter", Layers = "0:2")]
[Node("l{i}.down", "Gemm", "A=l{i}.act; B=l{i}.down; D=>l{i}.mlp[tokens,hidden]; M=tokens; N=hidden; K=inter; TransposeB=true; Alpha=1; Beta=0; WeightDType=F32", Layers = "0:2")]
[Node("l{i}.residual", "Residual", "A=h{i}; B=l{i}.mlp; Output=>h{i+1}[tokens,hidden]; Elements=tokens*hidden", Layers = "0:2")]
[Node("final_norm", "RmsNorm", "Input=h2; Weight=final_norm; Output=>final[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=final; Weight=lm_head; Logits=>logits[selected,vocab]; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class DenseModel
{
}

// The same model with the residual add and the next norm written as the fused AddRmsNorm. The generated forward must
// fall back to Residual + RmsNorm on a backend that lacks the fused op, and produce the same numbers either way.
[RoostModel("tiny-fused", Dtype = "F32")]
[Dim("hidden", 16)]
[Dim("inter", 32)]
[Dim("inter2", "2*inter")]
[Dim("vocab", 64)]
[Weight("embed", "vocab,hidden")]
[Weight("norm0", "hidden")]
[Weight("gate_up", "inter2,hidden")]
[Weight("down", "hidden,inter")]
[Weight("norm1", "hidden")]
[Weight("lm_head", "vocab,hidden")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
[Node("norm0", "RmsNorm", "Input=h0; Weight=norm0; Output=>x0[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("up", "Gemm", "A=x0; B=gate_up; D=>up[tokens,inter2]; M=tokens; N=inter2; K=hidden; TransposeB=true; Alpha=1; WeightDType=F32")]
[Node("act", "SiluAndMul", "Input=up; Output=>act[tokens,inter]; Rows=tokens; Dim=inter")]
[Node("down", "Gemm", "A=act; B=down; D=>mlp[tokens,hidden]; M=tokens; N=hidden; K=inter; TransposeB=true; Alpha=1; WeightDType=F32")]
[Node("add_norm", "AddRmsNorm", "Input=mlp=>normed; Residual=h0=>h1; Weight=norm1; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=normed; Weight=lm_head; Logits=>logits[selected,vocab]; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class FusedModel
{
}
