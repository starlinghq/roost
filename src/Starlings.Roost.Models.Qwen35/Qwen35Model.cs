using Starlings.Roost.ModelCompiler;

namespace Starlings.Roost.Models.Qwen35;

// Qwen3.5-0.8B (Qwen/Qwen3.5-0.8B at 2fc06364715b967f1860aea9cf38778875588b17; decision 15). 24 decoder layers: full attention at
// i % 4 == 3 (6 layers) and Gated DeltaNet elsewhere (18 layers), a SwiGLU MLP in every layer, the head tied to the embedding.
// Widths are from the checkpoint's config.json (text_config): hidden 1024, intermediate 3584, 8 query heads and 2 key/value
// heads of 256, 16 key and 16 value heads of 128 in the linear layers, convolution width 4, vocabulary 248320.
//
// Two state groups (C4): "attn" holds paged K and V for the six attention layers; "linear" holds, per linear layer, the float32
// delta-rule state [16, 128, 128] and the convolution state [6144, 3] in one recurrent slot. Weights are bfloat16; activations are
// bfloat16; the recurrent state is float32.
//
// Weight names and what a loader does with them are in Qwen35WeightMap. In short: gate_proj and up_proj are one slot ("a+b"),
// q_proj is two slots (the checkpoint interleaves each head's query and output gate), every RMS norm weight except the linear
// layers' gated norm is stored zero-centred and the slot holds 1 + w, and conv1d.weight [6144, 1, 4] is the slot [6144, 4].
//
// THE LINEAR-ATTENTION BLOCK: one fused in_proj_qkv produces q, k and v side by side; the causal convolution runs over all 6144
// channels and q, k and v are then read as row-strided column views of its output (GatedDeltaRuleArgs QStride, KStride, VStride,
// contract version 2), written Q=l{i}.c[0:key_dim] and so on.
//
// THE GATING STEPS (contract version 3, ops GatedDeltaRuleGates, GatedRmsNorm and SigmoidMul):
//   - linear layers: GatedDeltaRuleGates turns the raw projections a and b into g = -exp(A_log) * softplus(a + dt_bias) (float32)
//     and beta = sigmoid(b); the delta rule reads those. A_log (F32) and dt_bias (BF16) are weight slots.
//   - linear layers: GatedRmsNorm applies rmsnorm(core_out) * linear_norm.weight * silu(z) per value head (Rows = tokens * v_heads,
//     Dim = 128; the weight is F32 and not zero-centred) before out_proj.
//   - attention layers: q_proj interleaves each head's query and output-gate rows; the loader splits them into q_proj and
//     q_gate_proj, and SigmoidMul computes attn_out * sigmoid(gate) before o_proj.
//   - ZERO-CENTRED NORMS. Contract v3 has no zero-centred RmsNorm/QkNorm, so the decoder, q/k and final norm slots hold
//     bfloat16(1 + w) (Qwen35WeightMap OnePlus, the ADR's loading rule). transformers computes 1 + w in float32, so this drops the
//     low bits of 1 + w: parity of those norms is to bfloat16 weight rounding, not exact. The gated output norm is not zero-centred.
//
// Not modelled: the MTP head (Qwen35WeightMap.Mtp, unused in M0), vision, YaRN. Multimodal rope (mrope_interleaved=true, sections
// [11,11,10]) gives every section the same position for a text-only prompt, so it is the plain rotary embedding on the first 64
// dimensions. That is inferred from R-022's per-layer agreement at the attention layers and its bf16 emulation of layers 3
// and 15, not tested directly (docs/m0/roost-r022-cpu-parity.md).
//
// ATTENTION WINDOW: PagedAttentionArgs.WindowLeft is a window when it is not negative. The generator rejects a PagedAttention node
// that does not bind it (ROOST015), and CpuOps throws on a window of 0. Full attention binds WindowLeft=-1, no window (R-022 found
// the forward attending to the current token only when the field was left unbound). LmHead writes float32 logits (C3), so the
// logits value is declared F32.
[RoostModel("qwen3.5-0.8b", Dtype = "BF16")]
[Dim("hidden", 1024)]
[Dim("inter", 3584)]
[Dim("inter2", "2*inter")]
[Dim("vocab", 248320)]
[Dim("heads", 8)]
[Dim("kv_heads", 2)]
[Dim("head_dim", 256)]
[Dim("rotary_dim", 64)]
[Dim("q_dim", "heads*head_dim")]
[Dim("kv_dim", "kv_heads*head_dim")]
[Dim("k_heads", 16)]
[Dim("v_heads", 16)]
[Dim("k_dim", 128)]
[Dim("v_dim", 128)]
[Dim("key_dim", "k_heads*k_dim")]
[Dim("value_dim", "v_heads*v_dim")]
[Dim("conv_dim", "2*key_dim+value_dim")]
[Dim("conv_width", 4)]
[Dim("block_size", 16)]
[StateGroup("attn", "Full", Layers = "0:24%4=3", KvHeads = "kv_heads", HeadSize = "head_dim", BlockSize = "block_size")]
[StateGroup("linear", "Recurrent", Layers = "0:24%4!=3", StateShape = "v_heads,k_dim,v_dim", ConvShape = "conv_dim,conv_width-1", SnapshotSlots = "8")]
[Weight("embed", "vocab,hidden", Source = "model.language_model.embed_tokens.weight")]
[Weight("l{i}.input_norm", "hidden", Source = "model.language_model.layers.{i}.input_layernorm.weight", Layers = "0:24")]
[Weight("l{i}.q_proj", "q_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.q_proj.weight", Layers = "0:24%4=3")]
[Weight("l{i}.q_gate_proj", "q_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.q_proj.weight", Layers = "0:24%4=3")]
[Weight("l{i}.k_proj", "kv_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.k_proj.weight", Layers = "0:24%4=3")]
[Weight("l{i}.v_proj", "kv_dim,hidden", Source = "model.language_model.layers.{i}.self_attn.v_proj.weight", Layers = "0:24%4=3")]
[Weight("l{i}.q_norm", "head_dim", Source = "model.language_model.layers.{i}.self_attn.q_norm.weight", Layers = "0:24%4=3")]
[Weight("l{i}.k_norm", "head_dim", Source = "model.language_model.layers.{i}.self_attn.k_norm.weight", Layers = "0:24%4=3")]
[Weight("l{i}.o_proj", "hidden,q_dim", Source = "model.language_model.layers.{i}.self_attn.o_proj.weight", Layers = "0:24%4=3")]
[Weight("l{i}.in_proj_qkv", "conv_dim,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.conv", "conv_dim,conv_width", Source = "model.language_model.layers.{i}.linear_attn.conv1d.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.in_proj_z", "value_dim,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_z.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.in_proj_a", "v_heads,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_a.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.in_proj_b", "v_heads,hidden", Source = "model.language_model.layers.{i}.linear_attn.in_proj_b.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.A_log", "v_heads", Dtype = "F32", Source = "model.language_model.layers.{i}.linear_attn.A_log", Layers = "0:24%4!=3")]
[Weight("l{i}.dt_bias", "v_heads", Source = "model.language_model.layers.{i}.linear_attn.dt_bias", Layers = "0:24%4!=3")]
[Weight("l{i}.linear_norm", "v_dim", Dtype = "F32", Source = "model.language_model.layers.{i}.linear_attn.norm.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.out_proj", "hidden,value_dim", Source = "model.language_model.layers.{i}.linear_attn.out_proj.weight", Layers = "0:24%4!=3")]
[Weight("l{i}.post_norm", "hidden", Source = "model.language_model.layers.{i}.post_attention_layernorm.weight", Layers = "0:24")]
[Weight("l{i}.gate_up", "inter2,hidden", Source = "model.language_model.layers.{i}.mlp.gate_proj.weight+model.language_model.layers.{i}.mlp.up_proj.weight", Layers = "0:24")]
[Weight("l{i}.down", "hidden,inter", Source = "model.language_model.layers.{i}.mlp.down_proj.weight", Layers = "0:24")]
[Weight("final_norm", "hidden", Source = "model.language_model.norm.weight")]
[Node("embed", "Embedding", "Table=embed; Output=>h0[tokens,hidden]; Tokens=tokens; Dim=hidden; Vocab=vocab")]
// ---- one layer: token mixer (Gated DeltaNet or attention), then the MLP ----
[Node("l{i}.input_norm", "RmsNorm", "Input=h{i}; Weight=l{i}.input_norm; Output=>l{i}.x[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:24")]
// -- Gated DeltaNet --
[Node("l{i}.qkv", "Gemm", "A=l{i}.x; B=l{i}.in_proj_qkv; D=>l{i}.qkv[tokens,conv_dim]; M=tokens; N=conv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4!=3")]
[Node("l{i}.conv", "CausalConv1d", "Input=l{i}.qkv; Weight=l{i}.conv; Output=>l{i}.c[tokens,conv_dim]; ConvState=s:linear.conv; Batch=seqs; Tokens=tokens; Channels=conv_dim; Width=conv_width; Silu=true", Layers = "0:24%4!=3")]
[Node("l{i}.z", "Gemm", "A=l{i}.x; B=l{i}.in_proj_z; D=>l{i}.z[tokens,value_dim]; M=tokens; N=value_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4!=3")]
[Node("l{i}.a", "Gemm", "A=l{i}.x; B=l{i}.in_proj_a; D=>l{i}.a[tokens,v_heads]; M=tokens; N=v_heads; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4!=3")]
[Node("l{i}.b", "Gemm", "A=l{i}.x; B=l{i}.in_proj_b; D=>l{i}.b[tokens,v_heads]; M=tokens; N=v_heads; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4!=3")]
[Node("l{i}.gates", "GatedDeltaRuleGates", "A=l{i}.a; B=l{i}.b; ALog=l{i}.A_log; DtBias=l{i}.dt_bias; G=>l{i}.g[tokens,v_heads]:F32; Beta=>l{i}.beta[tokens,v_heads]; Tokens=tokens; VHeads=v_heads", Layers = "0:24%4!=3")]
[Node("l{i}.delta", "GatedDeltaRule", "Q=l{i}.c[0:key_dim]; K=l{i}.c[key_dim:2*key_dim]; V=l{i}.c[2*key_dim:conv_dim]; QStride=conv_dim; KStride=conv_dim; VStride=conv_dim; G=l{i}.g; Beta=l{i}.beta; Output=>l{i}.o[tokens,value_dim]; State=s:linear.state; Batch=seqs; Tokens=tokens; KHeads=k_heads; VHeads=v_heads; KDim=k_dim; VDim=v_dim; Scale=0.08838834764831845; UseInitialState=true; L2NormQk=true", Layers = "0:24%4!=3")]
[Node("l{i}.onorm", "GatedRmsNorm", "Input=l{i}.o; Gate=l{i}.z; Weight=l{i}.linear_norm; Output=>l{i}.on[tokens,value_dim]; Rows=tokens*v_heads; Dim=v_dim; Eps=1e-6", Layers = "0:24%4!=3")]
[Node("l{i}.out_proj", "Gemm", "A=l{i}.on; B=l{i}.out_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=value_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4!=3")]
// -- full attention --
[Node("l{i}.q", "Gemm", "A=l{i}.x; B=l{i}.q_proj; D=>l{i}.q[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.gate", "Gemm", "A=l{i}.x; B=l{i}.q_gate_proj; D=>l{i}.gate[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.k", "Gemm", "A=l{i}.x; B=l{i}.k_proj; D=>l{i}.k[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.v", "Gemm", "A=l{i}.x; B=l{i}.v_proj; D=>l{i}.v[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.qknorm", "QkNorm", "Q=l{i}.q=>l{i}.qn; K=l{i}.k=>l{i}.kn; QWeight=l{i}.q_norm; KWeight=l{i}.k_norm; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; Eps=1e-6", Layers = "0:24%4=3")]
[Node("l{i}.rope", "Rope", "Q=l{i}.qn=>l{i}.qr; K=l{i}.kn=>l{i}.kr; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; RotaryDim=rotary_dim; Kind=Default; Theta=10000000", Layers = "0:24%4=3")]
[Node("l{i}.cache", "CacheWrite", "K=l{i}.kr; V=l{i}.v; KCache=s:attn.k; VCache=s:attn.v; Tokens=tokens; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd", Layers = "0:24%4=3")]
[Node("l{i}.attn", "PagedAttention", "Q=l{i}.qr; KCache=s:attn.k; VCache=s:attn.v; Output=>l{i}.ao[tokens,q_dim]; Batch=seqs; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; Scale=0.0625; Causal=true; WindowLeft=-1", Layers = "0:24%4=3")]
[Node("l{i}.ogate", "SigmoidMul", "Input=l{i}.ao; Gate=l{i}.gate; Output=>l{i}.aog[tokens,q_dim]; Rows=tokens; Dim=q_dim", Layers = "0:24%4=3")]
[Node("l{i}.o_proj", "Gemm", "A=l{i}.aog; B=l{i}.o_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=q_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
// -- MLP --
[Node("l{i}.mix_res", "Residual", "A=h{i}; B=l{i}.mix; Output=>l{i}.hm[tokens,hidden]; Elements=tokens*hidden", Layers = "0:24")]
[Node("l{i}.post_norm", "RmsNorm", "Input=l{i}.hm; Weight=l{i}.post_norm; Output=>l{i}.y[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:24")]
[Node("l{i}.up", "Gemm", "A=l{i}.y; B=l{i}.gate_up; D=>l{i}.up[tokens,inter2]; M=tokens; N=inter2; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24")]
[Node("l{i}.act", "SiluAndMul", "Input=l{i}.up; Output=>l{i}.act[tokens,inter]; Rows=tokens; Dim=inter", Layers = "0:24")]
[Node("l{i}.down", "Gemm", "A=l{i}.act; B=l{i}.down; D=>l{i}.mlp[tokens,hidden]; M=tokens; N=hidden; K=inter; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24")]
[Node("l{i}.mlp_res", "Residual", "A=l{i}.hm; B=l{i}.mlp; Output=>h{i+1}[tokens,hidden]; Elements=tokens*hidden", Layers = "0:24")]
[Node("final_norm", "RmsNorm", "Input=h24; Weight=final_norm; Output=>final[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=final; Weight=embed; Logits=>logits[selected,vocab]:F32; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class Qwen35Model
{
    /// <summary>
    /// Whether <see cref="Forward{TOps}"/> computes the checkpoint's function: yes, on contract version 3 (the gating ops), checked per
    /// layer against transformers fp32 by R-022. One approximation remains and is named in <see cref="Approximations"/>.
    /// </summary>
    public const bool NumericallyComplete = true;

    /// <summary>Known deviations from transformers' arithmetic that the contract cannot remove yet.</summary>
    public static IReadOnlyList<string> Approximations { get; } =
    [
        "decoder, q/k and final RMS norms: the slots hold bfloat16(1 + w) (ADR); transformers computes 1 + w in float32",
    ];
}
