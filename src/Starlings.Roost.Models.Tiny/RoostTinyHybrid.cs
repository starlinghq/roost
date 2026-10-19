using Starlings.Roost.ModelCompiler;

namespace Starlings.Roost.Models.Tiny;

// roost-tiny-hybrid: a Qwen3.5-shaped hybrid decoder at CI size. Eight layers with full attention at i % 4 == 3 and
// Gated DeltaNet elsewhere (Qwen3.5-0.8B repeats the same pattern to 24 layers: 6 attention, 18 recurrent, decision 15),
// two state groups (paged K and V for the attention layers; recurrent state plus convolution state for the others),
// bf16 weights and activations, float32 recurrent state, and the head tied to the embedding like the checkpoint.
//
// THE LINEAR-ATTENTION PROJECTION IS FUSED, as in the checkpoint: one in_proj_qkv of [2*k_heads*k_dim + v_heads*v_dim, hidden]
// produces q, k and v side by side in every token row, and the convolution runs over all of those channels
// (the checkpoint's conv1d has 6144 channels = 2*2048 + 2048). The C3 ops take whole contiguous buffers, so the
// GatedDeltaRule that follows cannot read the q, k and v slices of that row: it needs views (a stream A change, the
// C3-qkv item). Until that lands, the three arguments Q, K and V of the "delta" nodes below are all bound to the whole
// convolution output. That keeps the weight slots, state groups, plan inputs, graph and call order exactly those of the
// real layout, and it makes the NUMERIC output of Forward wrong. Nothing may run this model's Forward on CpuOps and trust
// the logits; the structure tests do not, and HybridStructureTests fails when the placeholder is replaced, so that
// the switch to views cannot be forgotten. The Qwen3.5 head dimensions are narrowed here (16 where the checkpoint has 128).
//
// What this tiny model leaves out of Qwen3.5, because C3 has no op for it: the attention output gate (the checkpoint's
// q_proj is twice as wide), the linear block's z projection with its gated RMS norm, and the A_log and dt_bias
// transforms that turn in_proj_a and in_proj_b into the decay and the beta (here they are the raw projections). Rotary
// embedding covers a quarter of the head, as the checkpoint's partial_rotary_factor 0.25 does; multimodal rope sections
// are not modelled. Those are the real definition's work (R-035).
[RoostModel("roost-tiny-hybrid", Dtype = "BF16")]
[Dim("hidden", 64)]
[Dim("inter", 128)]
[Dim("inter2", "2*inter")]
[Dim("vocab", 256)]
[Dim("heads", 4)]
[Dim("kv_heads", 2)]
[Dim("head_dim", 16)]
[Dim("rotary_dim", 4)]
[Dim("q_dim", "heads*head_dim")]
[Dim("kv_dim", "kv_heads*head_dim")]
[Dim("k_heads", 4)]
[Dim("v_heads", 4)]
[Dim("k_dim", 16)]
[Dim("v_dim", 16)]
[Dim("conv_dim", "2*k_heads*k_dim+v_heads*v_dim")]
[Dim("conv_width", 4)]
[Dim("block_size", 16)]
[StateGroup("attn", "Full", Layers = "0:8%4=3", KvHeads = "kv_heads", HeadSize = "head_dim", BlockSize = "block_size")]
[StateGroup("linear", "Recurrent", Layers = "0:8%4!=3", StateShape = "v_heads,k_dim,v_dim", ConvShape = "conv_dim,conv_width-1", SnapshotSlots = "8")]
[Weight("embed", "vocab,hidden", Source = "model.language_model.embed_tokens.weight")]
[Weight("l{i}.input_norm", "hidden", Source = "model.language_model.layers.{i}.input_layernorm.weight", Layers = "0:8")]
[Weight("l{i}.q_proj", "q_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.q_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.k_proj", "kv_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.k_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.v_proj", "kv_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.v_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.q_norm", "head_dim", Source = "model.language_model.layers.{i}.self_attn.q_norm.weight", Layers = "0:8%4=3")]
[Weight("l{i}.k_norm", "head_dim", Source = "model.language_model.layers.{i}.self_attn.k_norm.weight", Layers = "0:8%4=3")]
[Weight("l{i}.o_proj", "hidden,q_dim", Source = "model.language_model.layers.{i}.self_attn.o_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.in_proj_qkv", "conv_dim,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.conv", "conv_dim,conv_width", Source = "model.language_model.layers.{i}.linear_attn.conv1d.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.in_proj_a", "v_heads,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_a.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.in_proj_b", "v_heads,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_b.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.out_proj", "hidden,v_heads*v_dim", Source = "model.language_model.layers.{i}.linear_attn.out_proj.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.post_norm", "hidden", Source = "model.language_model.layers.{i}.post_attention_layernorm.weight", Layers = "0:8")]
[Weight("l{i}.gate_up", "inter2,hidden", Source = "model.language_model.layers.{i}.mlp.gate_proj.weight+model.language_model.layers.{i}.mlp.up_proj.weight", Layers = "0:8")]
[Weight("l{i}.down", "hidden,inter", Source = "model.language_model.layers.{i}.mlp.down_proj.weight", Layers = "0:8")]
[Weight("final_norm", "hidden", Source = "model.language_model.norm.weight")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
// ---- one layer: token mixer (Gated DeltaNet or attention), then the MLP ----
[Node("l{i}.input_norm", "RmsNorm", "Input=h{i}; Weight=l{i}.input_norm; Output=>l{i}.x[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:8")]
[Node("l{i}.q", "Gemm", "A=l{i}.x; B=l{i}.q_proj; D=>l{i}.q[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.k", "Gemm", "A=l{i}.x; B=l{i}.k_proj; D=>l{i}.k[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.qkv", "Gemm", "A=l{i}.x; B=l{i}.in_proj_qkv; D=>l{i}.qkv[tokens,conv_dim]; M=tokens; N=conv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.conv", "CausalConv1d", "Input=l{i}.qkv; Weight=l{i}.conv; Output=>l{i}.qkvc[tokens,conv_dim]; ConvState=s:linear.conv; Batch=seqs; Tokens=tokens; Channels=conv_dim; Width=conv_width; Silu=true", Layers = "0:8%4!=3")]
[Node("l{i}.g", "Gemm", "A=l{i}.x; B=l{i}.in_proj_a; D=>l{i}.g[tokens,v_heads]; M=tokens; N=v_heads; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.beta", "Gemm", "A=l{i}.x; B=l{i}.in_proj_b; D=>l{i}.beta[tokens,v_heads]; M=tokens; N=v_heads; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.delta", "GatedDeltaRule", "Q=l{i}.qkvc; K=l{i}.qkvc; V=l{i}.qkvc; G=l{i}.g; Beta=l{i}.beta; Output=>l{i}.o[tokens,v_heads*v_dim]; State=s:linear.state; Batch=seqs; Tokens=tokens; KHeads=k_heads; VHeads=v_heads; KDim=k_dim; VDim=v_dim; Scale=0.25; UseInitialState=true; L2NormQk=true", Layers = "0:8%4!=3")]
[Node("l{i}.out_proj", "Gemm", "A=l{i}.o; B=l{i}.out_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=v_heads*v_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.v", "Gemm", "A=l{i}.x; B=l{i}.v_proj; D=>l{i}.v[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.qknorm", "QkNorm", "Q=l{i}.q=>l{i}.qn; K=l{i}.k=>l{i}.kn; QWeight=l{i}.q_norm; KWeight=l{i}.k_norm; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; Eps=1e-6", Layers = "0:8%4=3")]
[Node("l{i}.rope", "Rope", "Q=l{i}.qn=>l{i}.qr; K=l{i}.kn=>l{i}.kr; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; RotaryDim=rotary_dim; Kind=Default; Theta=10000000", Layers = "0:8%4=3")]
[Node("l{i}.cache", "CacheWrite", "K=l{i}.kr; V=l{i}.v; KCache=s:attn.k; VCache=s:attn.v; Tokens=tokens; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; KScale=1; VScale=1", Layers = "0:8%4=3")]
[Node("l{i}.attn", "PagedAttention", "Q=l{i}.qr; KCache=s:attn.k; VCache=s:attn.v; Output=>l{i}.ao[tokens,q_dim]; Batch=seqs; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; Scale=0.25; WindowLeft=-1; Causal=true", Layers = "0:8%4=3")]
[Node("l{i}.o_proj", "Gemm", "A=l{i}.ao; B=l{i}.o_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=q_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.mix_res", "Residual", "A=h{i}; B=l{i}.mix; Output=>l{i}.hm[tokens,hidden]; Elements=tokens*hidden", Layers = "0:8")]
[Node("l{i}.post_norm", "RmsNorm", "Input=l{i}.hm; Weight=l{i}.post_norm; Output=>l{i}.y[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:8")]
[Node("l{i}.up", "Gemm", "A=l{i}.y; B=l{i}.gate_up; D=>l{i}.up[tokens,inter2]; M=tokens; N=inter2; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8")]
[Node("l{i}.act", "SiluAndMul", "Input=l{i}.up; Output=>l{i}.act[tokens,inter]; Rows=tokens; Dim=inter", Layers = "0:8")]
[Node("l{i}.down", "Gemm", "A=l{i}.act; B=l{i}.down; D=>l{i}.mlp[tokens,hidden]; M=tokens; N=hidden; K=inter; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8")]
[Node("l{i}.mlp_res", "Residual", "A=l{i}.hm; B=l{i}.mlp; Output=>h{i+1}[tokens,hidden]; Elements=tokens*hidden", Layers = "0:8")]
[Node("final_norm", "RmsNorm", "Input=h8; Weight=final_norm; Output=>final[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=final; Weight=embed; Logits=>logits[selected,vocab]:F32; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class RoostTinyHybrid
{
}
