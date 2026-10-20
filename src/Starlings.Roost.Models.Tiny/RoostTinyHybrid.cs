using Starlings.Roost.ModelCompiler;

namespace Starlings.Roost.Models.Tiny;

// roost-tiny-hybrid: a Qwen3.5-shaped hybrid decoder at CI size. Eight layers with full attention at i % 4 == 3 and
// Gated DeltaNet elsewhere (Qwen3.5-0.8B repeats the same pattern to 24 layers: 6 attention, 18 recurrent, decision 15),
// two state groups (paged K and V for the attention layers; recurrent state plus convolution state for the others),
// bf16 weights and activations, float32 recurrent state, and the head tied to the embedding like the checkpoint.
//
// THE LINEAR-ATTENTION PROJECTION IS FUSED, as in the checkpoint: one in_proj_qkv of [2*k_heads*k_dim + v_heads*v_dim, hidden]
// produces q, k and v side by side in every token row, and the convolution runs over all of those channels (the checkpoint's
// conv1d has 6144 channels = 2*2048 + 2048). GatedDeltaRule reads q, k and v as three row-strided column views of the
// convolution output (contract version 2: QStride, KStride and VStride are conv_dim), written Q=l{i}.qkvc[0:key_dim] and so on.
//
// The Qwen3.5 gating of contract version 3 is in: GatedDeltaRuleGates turns the raw in_proj_a and in_proj_b outputs into the
// float32 log decay and the beta from A_log (float32) and dt_bias; the delta rule output goes through GatedRmsNorm with the
// in_proj_z output and linear_attn.norm (float32) before out_proj; the attention layers project the output gate with a second
// weight slot (q_gate_proj, as in R-035: the checkpoint's q_proj interleaves each head's query and gate rows, and the loader splits
// them) and apply SigmoidMul to the attention output before o_proj. Layer structure, slot names and node names follow
// Starlings.Roost.Models.Qwen35 (R-035), at CI size: head dimensions 16 where the checkpoint has 128 (linear) and 256 (attention).
//
// Not modelled, as in R-035's list: multimodal rope sections (partial rotary, a quarter of the head, is) and the zero-centred
// norm weights (the tiny norm weights are plain: 1 plus noise, applied as stored).
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
[Dim("key_dim", "k_heads*k_dim")]
[Dim("value_dim", "v_heads*v_dim")]
[Dim("conv_width", 4)]
[Dim("block_size", 16)]
[StateGroup("attn", "Full", Layers = "0:8%4=3", KvHeads = "kv_heads", HeadSize = "head_dim", BlockSize = "block_size")]
[StateGroup("linear", "Recurrent", Layers = "0:8%4!=3", StateShape = "v_heads,k_dim,v_dim", ConvShape = "conv_dim,conv_width-1", SnapshotSlots = "8")]
[Weight("embed", "vocab,hidden", Source = "model.language_model.embed_tokens.weight")]
[Weight("l{i}.input_norm", "hidden", Source = "model.language_model.layers.{i}.input_layernorm.weight", Layers = "0:8")]
[Weight("l{i}.q_proj", "q_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.q_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.q_gate_proj", "q_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.q_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.k_proj", "kv_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.k_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.v_proj", "kv_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.v_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.q_norm", "head_dim", Source = "model.language_model.layers.{i}.self_attn.q_norm.weight", Layers = "0:8%4=3")]
[Weight("l{i}.k_norm", "head_dim", Source = "model.language_model.layers.{i}.self_attn.k_norm.weight", Layers = "0:8%4=3")]
[Weight("l{i}.o_proj", "hidden,q_dim", Source = "model.language_model.layers.{i}.self_attn.o_proj.weight", Layers = "0:8%4=3")]
[Weight("l{i}.in_proj_qkv", "conv_dim,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.conv", "conv_dim,conv_width", Source = "model.language_model.layers.{i}.linear_attn.conv1d.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.in_proj_z", "value_dim,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_z.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.in_proj_a", "v_heads,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_a.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.in_proj_b", "v_heads,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_b.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.A_log", "v_heads", Dtype = "F32", Source = "model.language_model.layers.{i}.linear_attn.A_log", Layers = "0:8%4!=3")]
[Weight("l{i}.dt_bias", "v_heads", Source = "model.language_model.layers.{i}.linear_attn.dt_bias", Layers = "0:8%4!=3")]
[Weight("l{i}.linear_norm", "v_dim", Dtype = "F32", Source = "model.language_model.layers.{i}.linear_attn.norm.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.out_proj", "hidden,value_dim", Source = "model.language_model.layers.{i}.linear_attn.out_proj.weight", Layers = "0:8%4!=3")]
[Weight("l{i}.post_norm", "hidden", Source = "model.language_model.layers.{i}.post_attention_layernorm.weight", Layers = "0:8")]
[Weight("l{i}.gate_up", "inter2,hidden", Source = "model.language_model.layers.{i}.mlp.gate_proj.weight+model.language_model.layers.{i}.mlp.up_proj.weight", Layers = "0:8")]
[Weight("l{i}.down", "hidden,inter", Source = "model.language_model.layers.{i}.mlp.down_proj.weight", Layers = "0:8")]
[Weight("final_norm", "hidden", Source = "model.language_model.norm.weight")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
// ---- one layer: token mixer (Gated DeltaNet or attention), then the MLP ----
[Node("l{i}.input_norm", "RmsNorm", "Input=h{i}; Weight=l{i}.input_norm; Output=>l{i}.x[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:8")]
[Node("l{i}.q", "Gemm", "A=l{i}.x; B=l{i}.q_proj; D=>l{i}.q[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.gate", "Gemm", "A=l{i}.x; B=l{i}.q_gate_proj; D=>l{i}.gate[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.k", "Gemm", "A=l{i}.x; B=l{i}.k_proj; D=>l{i}.k[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.qkv", "Gemm", "A=l{i}.x; B=l{i}.in_proj_qkv; D=>l{i}.qkv[tokens,conv_dim]; M=tokens; N=conv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.conv", "CausalConv1d", "Input=l{i}.qkv; Weight=l{i}.conv; Output=>l{i}.qkvc[tokens,conv_dim]; ConvState=s:linear.conv; Batch=seqs; Tokens=tokens; Channels=conv_dim; Width=conv_width; Silu=true", Layers = "0:8%4!=3")]
[Node("l{i}.z", "Gemm", "A=l{i}.x; B=l{i}.in_proj_z; D=>l{i}.z[tokens,value_dim]; M=tokens; N=value_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.a", "Gemm", "A=l{i}.x; B=l{i}.in_proj_a; D=>l{i}.a[tokens,v_heads]; M=tokens; N=v_heads; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.b", "Gemm", "A=l{i}.x; B=l{i}.in_proj_b; D=>l{i}.b[tokens,v_heads]; M=tokens; N=v_heads; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.gates", "GatedDeltaRuleGates", "A=l{i}.a; B=l{i}.b; ALog=l{i}.A_log; DtBias=l{i}.dt_bias; G=>l{i}.g[tokens,v_heads]:F32; Beta=>l{i}.beta[tokens,v_heads]; Tokens=tokens; VHeads=v_heads", Layers = "0:8%4!=3")]
[Node("l{i}.delta", "GatedDeltaRule", "Q=l{i}.qkvc[0:key_dim]; K=l{i}.qkvc[key_dim:2*key_dim]; V=l{i}.qkvc[2*key_dim:conv_dim]; QStride=conv_dim; KStride=conv_dim; VStride=conv_dim; G=l{i}.g; Beta=l{i}.beta; Output=>l{i}.o[tokens,value_dim]; State=s:linear.state; Batch=seqs; Tokens=tokens; KHeads=k_heads; VHeads=v_heads; KDim=k_dim; VDim=v_dim; Scale=0.25; UseInitialState=true; L2NormQk=true", Layers = "0:8%4!=3")]
[Node("l{i}.onorm", "GatedRmsNorm", "Input=l{i}.o; Gate=l{i}.z; Weight=l{i}.linear_norm; Output=>l{i}.on[tokens,value_dim]; Rows=tokens*v_heads; Dim=v_dim; Eps=1e-6", Layers = "0:8%4!=3")]
[Node("l{i}.out_proj", "Gemm", "A=l{i}.on; B=l{i}.out_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=value_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4!=3")]
[Node("l{i}.v", "Gemm", "A=l{i}.x; B=l{i}.v_proj; D=>l{i}.v[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
[Node("l{i}.qknorm", "QkNorm", "Q=l{i}.q=>l{i}.qn; K=l{i}.k=>l{i}.kn; QWeight=l{i}.q_norm; KWeight=l{i}.k_norm; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; Eps=1e-6", Layers = "0:8%4=3")]
[Node("l{i}.rope", "Rope", "Q=l{i}.qn=>l{i}.qr; K=l{i}.kn=>l{i}.kr; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; RotaryDim=rotary_dim; Kind=Default; Theta=10000000", Layers = "0:8%4=3")]
[Node("l{i}.cache", "CacheWrite", "K=l{i}.kr; V=l{i}.v; KCache=s:attn.k; VCache=s:attn.v; Tokens=tokens; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; KScale=1; VScale=1", Layers = "0:8%4=3")]
[Node("l{i}.attn", "PagedAttention", "Q=l{i}.qr; KCache=s:attn.k; VCache=s:attn.v; Output=>l{i}.ao[tokens,q_dim]; Batch=seqs; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; Scale=0.25; WindowLeft=-1; Causal=true", Layers = "0:8%4=3")]
[Node("l{i}.agate", "SigmoidMul", "Input=l{i}.ao; Gate=l{i}.gate; Output=>l{i}.aog[tokens,q_dim]; Rows=tokens; Dim=q_dim", Layers = "0:8%4=3")]
[Node("l{i}.o_proj", "Gemm", "A=l{i}.aog; B=l{i}.o_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=q_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:8%4=3")]
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
