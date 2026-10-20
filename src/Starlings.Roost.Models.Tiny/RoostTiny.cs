using Starlings.Roost.ModelCompiler;

namespace Starlings.Roost.Models.Tiny;

// roost-tiny: a Qwen3-shaped dense decoder at CI size. Every layer is grouped-query attention with per-head Q and K
// RMS norms (Qwen3's q_norm and k_norm), NeoX rotary embedding and a SwiGLU MLP; the head is not tied to the embedding
// (Qwen3-1.7B ties it, larger Qwen3 models do not; the hybrid model covers the tied form). One full-attention state group.
// float32 activations and weights so the CPU reference compares tightly; the checkpoint-shaped bf16 form is the hybrid's.
// Widths: hidden 64, 4 query heads over 2 KV heads of 16, MLP 128, vocabulary 256, 4 layers. TinyConfig.Dense repeats
// these numbers and a test compares them with the generated slots.
//
// The checkpoint names in Source follow the Qwen3 safetensors layout. gate_up is the loader's row-wise concatenation of
// gate_proj then up_proj, which is the order SiluAndMul reads; the "+" joins the two checkpoint tensors.
[RoostModel("roost-tiny", Dtype = "F32")]
[Dim("hidden", 64)]
[Dim("inter", 128)]
[Dim("inter2", "2*inter")]
[Dim("vocab", 256)]
[Dim("heads", 4)]
[Dim("kv_heads", 2)]
[Dim("head_dim", 16)]
[Dim("q_dim", "heads*head_dim")]
[Dim("kv_dim", "kv_heads*head_dim")]
[Dim("block_size", 16)]
[StateGroup("attn", "Full", Layers = "0:4", KvHeads = "kv_heads", HeadSize = "head_dim", BlockSize = "block_size")]
[Weight("embed", "vocab,hidden", Source = "model.embed_tokens.weight")]
[Weight("l{i}.input_norm", "hidden", Source = "model.layers.{i}.input_layernorm.weight", Layers = "0:4")]
[Weight("l{i}.q_proj", "q_dim,hidden", Source = "model.layers.{i}.self_attn.q_proj.weight", Layers = "0:4")]
[Weight("l{i}.k_proj", "kv_dim,hidden", Source = "model.layers.{i}.self_attn.k_proj.weight", Layers = "0:4")]
[Weight("l{i}.v_proj", "kv_dim,hidden", Source = "model.layers.{i}.self_attn.v_proj.weight", Layers = "0:4")]
[Weight("l{i}.q_norm", "head_dim", Source = "model.layers.{i}.self_attn.q_norm.weight", Layers = "0:4")]
[Weight("l{i}.k_norm", "head_dim", Source = "model.layers.{i}.self_attn.k_norm.weight", Layers = "0:4")]
[Weight("l{i}.o_proj", "hidden,q_dim", Source = "model.layers.{i}.self_attn.o_proj.weight", Layers = "0:4")]
[Weight("l{i}.post_norm", "hidden", Source = "model.layers.{i}.post_attention_layernorm.weight", Layers = "0:4")]
[Weight("l{i}.gate_up", "inter2,hidden", Source = "model.layers.{i}.mlp.gate_proj.weight+model.layers.{i}.mlp.up_proj.weight", Layers = "0:4")]
[Weight("l{i}.down", "hidden,inter", Source = "model.layers.{i}.mlp.down_proj.weight", Layers = "0:4")]
[Weight("final_norm", "hidden", Source = "model.norm.weight")]
[Weight("lm_head", "vocab,hidden", Source = "lm_head.weight")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
[Node("l{i}.input_norm", "RmsNorm", "Input=h{i}; Weight=l{i}.input_norm; Output=>l{i}.x[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:4")]
[Node("l{i}.q", "Gemm", "A=l{i}.x; B=l{i}.q_proj; D=>l{i}.q[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=F32", Layers = "0:4")]
[Node("l{i}.k", "Gemm", "A=l{i}.x; B=l{i}.k_proj; D=>l{i}.k[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=F32", Layers = "0:4")]
[Node("l{i}.v", "Gemm", "A=l{i}.x; B=l{i}.v_proj; D=>l{i}.v[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=F32", Layers = "0:4")]
[Node("l{i}.qknorm", "QkNorm", "Q=l{i}.q=>l{i}.qn; K=l{i}.k=>l{i}.kn; QWeight=l{i}.q_norm; KWeight=l{i}.k_norm; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; Eps=1e-6", Layers = "0:4")]
[Node("l{i}.rope", "Rope", "Q=l{i}.qn=>l{i}.qr; K=l{i}.kn=>l{i}.kr; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; RotaryDim=head_dim; Kind=Default; Theta=1000000", Layers = "0:4")]
[Node("l{i}.cache", "CacheWrite", "K=l{i}.kr; V=l{i}.v; KCache=s:attn.k; VCache=s:attn.v; Tokens=tokens; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; KScale=1; VScale=1", Layers = "0:4")]
[Node("l{i}.attn", "PagedAttention", "Q=l{i}.qr; KCache=s:attn.k; VCache=s:attn.v; Output=>l{i}.ao[tokens,q_dim]; Batch=seqs; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; Scale=0.25; WindowLeft=-1; Causal=true", Layers = "0:4")]
[Node("l{i}.o_proj", "Gemm", "A=l{i}.ao; B=l{i}.o_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=q_dim; TransposeB=true; Alpha=1; WeightDType=F32", Layers = "0:4")]
[Node("l{i}.mix_res", "Residual", "A=h{i}; B=l{i}.mix; Output=>l{i}.hm[tokens,hidden]; Elements=tokens*hidden", Layers = "0:4")]
[Node("l{i}.post_norm", "RmsNorm", "Input=l{i}.hm; Weight=l{i}.post_norm; Output=>l{i}.y[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:4")]
[Node("l{i}.up", "Gemm", "A=l{i}.y; B=l{i}.gate_up; D=>l{i}.up[tokens,inter2]; M=tokens; N=inter2; K=hidden; TransposeB=true; Alpha=1; WeightDType=F32", Layers = "0:4")]
[Node("l{i}.act", "SiluAndMul", "Input=l{i}.up; Output=>l{i}.act[tokens,inter]; Rows=tokens; Dim=inter", Layers = "0:4")]
[Node("l{i}.down", "Gemm", "A=l{i}.act; B=l{i}.down; D=>l{i}.mlp[tokens,hidden]; M=tokens; N=hidden; K=inter; TransposeB=true; Alpha=1; WeightDType=F32", Layers = "0:4")]
[Node("l{i}.mlp_res", "Residual", "A=l{i}.hm; B=l{i}.mlp; Output=>h{i+1}[tokens,hidden]; Elements=tokens*hidden", Layers = "0:4")]
[Node("final_norm", "RmsNorm", "Input=h4; Weight=final_norm; Output=>final[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=final; Weight=lm_head; Logits=>logits[selected,vocab]:F32; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class RoostTiny
{
}
