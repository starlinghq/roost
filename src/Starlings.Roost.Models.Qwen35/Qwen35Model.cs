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
// WHAT THIS FORWARD DOES NOT YET COMPUTE. C3 has no op for four steps of the Qwen3.5 blocks, so the logits of this forward are
// NOT those of the checkpoint. Qwen35Model.NumericallyComplete is false until the ops exist; the definition already holds the
// projections and weights these steps need so each lands as one node:
//   1. linear layers: beta = sigmoid(b) and g = -exp(A_log) * softplus(a + dt_bias). The delta rule gets the raw projections a
//      and b as G and Beta. (A_log, dt_bias are reserved slots.)
//   2. linear layers: the output norm rmsnorm(core_out) * norm.weight * silu(z), per value head. out_proj reads the
//      delta-rule output directly. (z is projected but unread; norm.weight is a reserved slot.)
//   3. attention layers: the output gate, attn_out * sigmoid(gate). o_proj reads the attention output directly. (The gate is
//      projected by q_gate_proj but unread.)
//   4. RmsNorm and QkNorm have no zero-centred form, so the slots hold bfloat16(1 + w), which loses the low bits of 1 + w that
//      transformers keeps in float32.
// The contract-change request in the R-035 report names the ops. Nothing may trust these logits for parity (R-022 needs them).
//
// Not modelled: the MTP head (Qwen35WeightMap.Mtp, unused in M0), vision, multimodal rope sections (a text-only prompt makes the
// three position rows equal, which is plain rotary embedding), YaRN.
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
[Weight("l{i}.A_log", "v_heads", Dtype = "F32", Source = "model.language_model.layers.{i}.linear_attn.A_log", Layers = "0:24%4!=3", Reserved = true)]
[Weight("l{i}.dt_bias", "v_heads", Source = "model.language_model.layers.{i}.linear_attn.dt_bias", Layers = "0:24%4!=3", Reserved = true)]
[Weight("l{i}.linear_norm", "v_dim", Dtype = "F32", Source = "model.language_model.layers.{i}.linear_attn.norm.weight", Layers = "0:24%4!=3", Reserved = true)]
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
[Node("l{i}.delta", "GatedDeltaRule", "Q=l{i}.c[0:key_dim]; K=l{i}.c[key_dim:2*key_dim]; V=l{i}.c[2*key_dim:conv_dim]; QStride=conv_dim; KStride=conv_dim; VStride=conv_dim; G=l{i}.a; Beta=l{i}.b; Output=>l{i}.o[tokens,value_dim]; State=s:linear.state; Batch=seqs; Tokens=tokens; KHeads=k_heads; VHeads=v_heads; KDim=k_dim; VDim=v_dim; Scale=0.08838834764831845; UseInitialState=true; L2NormQk=true", Layers = "0:24%4!=3")]
[Node("l{i}.out_proj", "Gemm", "A=l{i}.o; B=l{i}.out_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=value_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4!=3")]
// -- full attention --
[Node("l{i}.q", "Gemm", "A=l{i}.x; B=l{i}.q_proj; D=>l{i}.q[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.gate", "Gemm", "A=l{i}.x; B=l{i}.q_gate_proj; D=>l{i}.gate[tokens,q_dim]; M=tokens; N=q_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.k", "Gemm", "A=l{i}.x; B=l{i}.k_proj; D=>l{i}.k[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.v", "Gemm", "A=l{i}.x; B=l{i}.v_proj; D=>l{i}.v[tokens,kv_dim]; M=tokens; N=kv_dim; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
[Node("l{i}.qknorm", "QkNorm", "Q=l{i}.q=>l{i}.qn; K=l{i}.k=>l{i}.kn; QWeight=l{i}.q_norm; KWeight=l{i}.k_norm; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; Eps=1e-6", Layers = "0:24%4=3")]
[Node("l{i}.rope", "Rope", "Q=l{i}.qn=>l{i}.qr; K=l{i}.kn=>l{i}.kr; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; RotaryDim=rotary_dim; Kind=Default; Theta=10000000", Layers = "0:24%4=3")]
[Node("l{i}.cache", "CacheWrite", "K=l{i}.kr; V=l{i}.v; KCache=s:attn.k; VCache=s:attn.v; Tokens=tokens; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd", Layers = "0:24%4=3")]
[Node("l{i}.attn", "PagedAttention", "Q=l{i}.qr; KCache=s:attn.k; VCache=s:attn.v; Output=>l{i}.ao[tokens,q_dim]; Batch=seqs; Tokens=tokens; QHeads=heads; KvHeads=kv_heads; HeadDim=head_dim; BlockSize=block_size; KvLayout=PagedNhd; Scale=0.0625; Causal=true", Layers = "0:24%4=3")]
[Node("l{i}.o_proj", "Gemm", "A=l{i}.ao; B=l{i}.o_proj; D=>l{i}.mix[tokens,hidden]; M=tokens; N=hidden; K=q_dim; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24%4=3")]
// -- MLP --
[Node("l{i}.mix_res", "Residual", "A=h{i}; B=l{i}.mix; Output=>l{i}.hm[tokens,hidden]; Elements=tokens*hidden", Layers = "0:24")]
[Node("l{i}.post_norm", "RmsNorm", "Input=l{i}.hm; Weight=l{i}.post_norm; Output=>l{i}.y[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6", Layers = "0:24")]
[Node("l{i}.up", "Gemm", "A=l{i}.y; B=l{i}.gate_up; D=>l{i}.up[tokens,inter2]; M=tokens; N=inter2; K=hidden; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24")]
[Node("l{i}.act", "SiluAndMul", "Input=l{i}.up; Output=>l{i}.act[tokens,inter]; Rows=tokens; Dim=inter", Layers = "0:24")]
[Node("l{i}.down", "Gemm", "A=l{i}.act; B=l{i}.down; D=>l{i}.mlp[tokens,hidden]; M=tokens; N=hidden; K=inter; TransposeB=true; Alpha=1; WeightDType=BF16", Layers = "0:24")]
[Node("l{i}.mlp_res", "Residual", "A=l{i}.hm; B=l{i}.mlp; Output=>h{i+1}[tokens,hidden]; Elements=tokens*hidden", Layers = "0:24")]
[Node("final_norm", "RmsNorm", "Input=h24; Weight=final_norm; Output=>final[tokens,hidden]; Rows=tokens; Dim=hidden; Eps=1e-6")]
[Node("head", "LmHead", "Hidden=final; Weight=embed; Logits=>logits[selected,vocab]; Selected=selected; Dim=hidden; Vocab=vocab; VocabPadded=vocab")]
[Output("logits")]
public static partial class Qwen35Model
{
    /// <summary>
    /// Whether <see cref="Forward{TOps}"/> computes the checkpoint's function. False: C3 lacks the ops for the beta and decay
    /// of the linear layers, their gated output norm and the attention output gate (see <see cref="PendingOps"/>).
    /// </summary>
    public const bool NumericallyComplete = false;

    /// <summary>The steps of Qwen3.5 the forward leaves out because C3 has no op for them, with where each belongs.</summary>
    public static IReadOnlyList<string> PendingOps { get; } =
    [
        "linear layers, before the delta rule: beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)",
        "linear layers, after the delta rule: rmsnorm(core_out) * linear_norm.weight * silu(z), per value head",
        "attention layers, after attention: attn_out * sigmoid(gate)",
        "all RMS norms: zero-centred weights (output * (1 + w)) without the bfloat16 rounding of 1 + w",
    ];
}
