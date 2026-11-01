using Starlings.Roost.ModelCompiler;

namespace Starlings.Roost.Models.Tiny;

// roost-tiny-moe: a Qwen3-MoE-shaped decoder at CI size (R-080). The attention half of every layer is roost-tiny's (grouped-query
// attention with per-head Q and K RMS norms, NeoX rope at theta 1e6, one full-attention state group); the MLP of every layer is a
// routed-expert block as in Qwen3-30B-A3B (decoder_sparse_step 1, no mlp_only_layers, no shared expert, norm_topk_prob true):
// a router Gemm to [tokens, experts] logits, then FusedMoe, which CpuOps defines as softmax, top-k by logit with ties to the lower
// expert, renormalised float32 weights, stable permutation, a grouped gate+up Gemm, SiluAndMul, a grouped down Gemm and the weighted
// combine. bf16 weights and activations like the checkpoint; the head is not tied (Qwen3-30B-A3B ties nothing). TinyConfig.Moe repeats
// the widths and a test compares them with the generated slots: hidden 64, 4 query heads over 2 KV heads of 16, vocabulary 256, 4 layers,
// 8 experts with 2 per token, expert width 32.
//
// Expert weights are stacked per layer: experts_gate_up is [experts, 2 * moe_inter, hidden] (each expert's gate_proj rows then its
// up_proj rows, the order SiluAndMul reads) and experts_down is [experts, hidden, moe_inter]. In Source, "*" stands for every expert
// index in ascending order and "+" joins the gate and up tensors of one expert; that stacking convention is a proposal for the loader,
// which does not read MoE checkpoints yet. FP8 and INT4 expert weights are unsupported here (R-077).
//
// The unfused form (MoeRouteTopK, MoePermute, GroupedGemm, SiluAndMul, GroupedGemm, MoeUnpermute as separate nodes) cannot be written
// as a definition today: the permuted rows are [tokens * top_k, hidden], and ExpertCounts and ExpertOffsets are [experts] and
// [experts + 1] int32, while the model compiler lays out only activations whose leading dimension is tokens or selected.
[RoostModel("roost-tiny-moe", Dtype = "BF16")]
[Dim("hidden", 64)]
[Dim("vocab", 256)]
[Dim("heads", 4)]
[Dim("kv_heads", 2)]
[Dim("head_dim", 16)]
[Dim("q_dim", "heads*head_dim")]
[Dim("kv_dim", "kv_heads*head_dim")]
[Dim("experts", 8)]
[Dim("top_k", 2)]
[Dim("moe_inter", 32)]
[Dim("moe_inter2", "2*moe_inter")]
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
[Weight("l{i}.router", "experts,hidden", Source = "model.layers.{i}.mlp.gate.weight", Layers = "0:4")]
[Weight("l{i}.experts_gate_up", "experts,moe_inter2,hidden", Source = "model.layers.{i}.mlp.experts.*.gate_proj.weight+model.layers.{i}.mlp.experts.*.up_proj.weight", Layers = "0:4")]
[Weight("l{i}.experts_down", "experts,hidden,moe_inter", Source = "model.layers.{i}.mlp.experts.*.down_proj.weight", Layers = "0:4")]
[Weight("final_norm", "hidden", Source = "model.norm.weight")]
[Weight("lm_head", "vocab,hidden", Source = "lm_head.weight")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
[Node("l{i}.input_norm", "RmsNorm", "Input=h{i}; Weight=l{i}.input_norm; Output=>l{i}.x[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:4")]
[Node("l{i}.q", "Gemm", "A=l{i}.x; B=l{i}.q_proj; D=>l{i}.q[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:4")]
[Node("l{i}.k", "Gemm", "A=l{i}.x; B=l{i}.k_proj; D=>l{i}.k[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:4")]
[Node("l{i}.v", "Gemm", "A=l{i}.x; B=l{i}.v_proj; D=>l{i}.v[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:4")]
[Node("l{i}.qknorm", "QkNorm", "Q=l{i}.q=>l{i}.qn; K=l{i}.k=>l{i}.kn; QWeight=l{i}.q_norm; KWeight=l{i}.k_norm; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; Eps=1e-6", Layers = "0:4")]
[Node("l{i}.rope", "Rope", "Q=l{i}.qn=>l{i}.qr; K=l{i}.kn=>l{i}.kr; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; RotaryDim=head_dim; Kind=Default; Theta=1000000", Layers = "0:4")]
[Node("l{i}.cache", "CacheWrite", "K=l{i}.kr; V=l{i}.v; KCache=s:attn.k; VCache=s:attn.v; Tokens=tokens; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; KScale=1; VScale=1", Layers = "0:4")]
[Node("l{i}.attn", "PagedAttention", "Q=l{i}.qr; KCache=s:attn.k; VCache=s:attn.v; Output=>l{i}.ao[tokens,q_dim]; Batch=seqs; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; Scale=0.25; WindowLeft=-1; Causal=true", Layers = "0:4")]
[Node("l{i}.o_proj", "Gemm", "A=l{i}.ao; B=l{i}.o_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=q_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:4")]
[Node("l{i}.mix_res", "Residual", "A=h{i}; B=l{i}.mix; Output=>l{i}.hm[tokens,hidden]; Elements=tokens*hidden", Layers = "0:4")]
[Node("l{i}.post_norm", "RmsNorm", "Input=l{i}.hm; Weight=l{i}.post_norm; Output=>l{i}.y[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:4")]
[Node("l{i}.router", "Gemm", "A=l{i}.y; B=l{i}.router; D=>l{i}.router_logits[tokens,experts]; M=tokens; N=experts; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:4")]
[Node("l{i}.moe", "FusedMoe", "Input=l{i}.y; RouterLogits=l{i}.router_logits; W13=l{i}.experts_gate_up; W2=l{i}.experts_down; Output=>l{i}.mlp[tokens,hidden]; Tokens=tokens; Dim=hidden; Intermediate=moe_inter; Experts=experts; TopK=top_k; Renormalize=true; WeightDType=BF16", Layers = "0:4")]
[Node("l{i}.mlp_res", "Residual", "A=l{i}.hm; B=l{i}.mlp; Output=>h{i+1}[tokens,hidden]; Elements=tokens*hidden", Layers = "0:4")]
[Node("final_norm", "RmsNorm", "Input=h4; Weight=final_norm; Output=>final[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=final; Weight=lm_head; Logits=>logits[selected,vocab]:F32; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class RoostTinyMoe
{
}
