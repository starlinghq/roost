using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops;

/// <summary>
/// The op vocabulary (contract C3): every op a model forward may call, as static abstract members so that
/// <c>Forward&lt;TOps&gt;() where TOps : struct, IRoostOps</c> is specialised and devirtualised under the JIT and
/// Native AOT (build plan section 3.1).
/// </summary>
/// <remarks>
/// <para>M0 ops are <c>static abstract</c>: every backend implements them.</para>
/// <para>
/// Optional fused ops are <c>static virtual</c> with a <c>SupportsX</c> property that defaults to false. Each is defined
/// as its unfused parts; the generated forward tests <c>TOps.SupportsX</c> and emits the fused call or the unfused
/// sequence (S4 finding 1: a static virtual default body cannot call the interface's other static members without a
/// self type parameter). The CRTP form <c>IRoostOps&lt;TSelf&gt;</c> was not measured and is not used.
/// </para>
/// <para>
/// Ops reserved for later phases are declared now so adding them never changes the interface: they are
/// <c>static virtual</c>, throw <see cref="NotSupportedException"/> by default, and report
/// <see cref="Supports"/> false until a backend overrides them when their item lands.
/// </para>
/// <para>Calls never allocate managed memory and never synchronise; a backend that runs on a stream is told it by its own state.</para>
/// </remarks>
public interface IRoostOps
{
    /// <summary>Whether this backend implements <paramref name="op"/>. M0 ops are always implemented.</summary>
    /// <param name="op">The op.</param>
    /// <returns>True when calling the op does not throw <see cref="NotSupportedException"/>.</returns>
    static virtual bool Supports(OpId op) => OpCatalog.IsM0(op);

    /// <summary><c>Output = Input * rsqrt(mean(Input^2) + Eps) * Weight</c>, per row. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void RmsNorm(in RmsNormArgs a);

    /// <summary>Per-head RmsNorm of query and key heads, in place. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void QkNorm(in QkNormArgs a);

    /// <summary>Rotary position embedding of Q and K in place: default or llama3 scaling. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void Rope(in RopeArgs a);

    /// <summary>Paged attention over one state group's KV pools, decode or prefill. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void PagedAttention(in PagedAttentionArgs a);

    /// <summary><c>Output = silu(Input[:, :Dim]) * Input[:, Dim:]</c>. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void SiluAndMul(in SiluAndMulArgs a);

    /// <summary><c>Output = gelu(Input[:, :Dim]) * Input[:, Dim:]</c>. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void GeluAndMul(in GeluAndMulArgs a);

    /// <summary><c>D = Alpha * A @ op(B) + Beta * C (+ Bias)</c> with fp32 accumulation. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void Gemm(in GemmArgs a);

    /// <summary><c>Output[i] = Table[Ids[i]]</c>. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void Embedding(in EmbeddingArgs a);

    /// <summary>Logits for the selected rows: <c>Logits = Hidden[Rows] @ Weight^T</c>. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void LmHead(in LmHeadArgs a);

    /// <summary><c>Output = A + B</c>; A may alias Output. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void Residual(in ResidualArgs a);

    /// <summary>Write new K and V rows into the paged pools. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void CacheWrite(in CacheWriteArgs a);

    /// <summary>Copy whole blocks within or between pools (copy-on-write forks, compaction, tier moves). M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void CacheCopy(in CacheCopyArgs a);

    /// <summary><c>Ids[r] = argmax(Logits[r, :Vocab])</c>, ties to the lowest index. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void Argmax(in ArgmaxArgs a);

    /// <summary>Temperature, top-k, top-p and min-p sampling with per-row parameters, seeds and optional token masks. M0.</summary>
    /// <param name="a">Arguments.</param>
    static abstract void Sample(in SampleArgs a);

    /// <summary>Whether <see cref="AddRmsNorm"/> is implemented; when false the forward calls its unfused parts.</summary>
    static virtual bool SupportsAddRmsNorm => false;

    /// <summary>Optional fused op: Residual then RmsNorm: <c>Residual += Input; Input = RmsNorm(Residual)</c>.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">The backend does not provide the fused form.</exception>
    static virtual void AddRmsNorm(in AddRmsNormArgs a) => throw new NotSupportedException(nameof(AddRmsNorm));

    /// <summary>Whether <see cref="AddRmsNormQuant"/> is implemented; when false the forward calls its unfused parts.</summary>
    static virtual bool SupportsAddRmsNormQuant => false;

    /// <summary>Optional fused op: AddRmsNorm then Quantize of the normalised output.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">The backend does not provide the fused form.</exception>
    static virtual void AddRmsNormQuant(in AddRmsNormQuantArgs a) => throw new NotSupportedException(nameof(AddRmsNormQuant));

    /// <summary>Whether <see cref="SiluMulQuant"/> is implemented; when false the forward calls its unfused parts.</summary>
    static virtual bool SupportsSiluMulQuant => false;

    /// <summary>Optional fused op: SiluAndMul then Quantize.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">The backend does not provide the fused form.</exception>
    static virtual void SiluMulQuant(in SiluMulQuantArgs a) => throw new NotSupportedException(nameof(SiluMulQuant));

    /// <summary>Whether <see cref="QkvQkNormRopeCacheWrite"/> is implemented; when false the forward calls its unfused parts.</summary>
    static virtual bool SupportsQkvQkNormRopeCacheWrite => false;

    /// <summary>Optional fused op: Split of a fused QKV projection, QkNorm, Rope and CacheWrite.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">The backend does not provide the fused form.</exception>
    static virtual void QkvQkNormRopeCacheWrite(in QkvQkNormRopeCacheWriteArgs a) => throw new NotSupportedException(nameof(QkvQkNormRopeCacheWrite));

    /// <summary>Whether <see cref="GemmLoraEpilogue"/> is implemented; when false the forward calls its unfused parts.</summary>
    static virtual bool SupportsGemmLoraEpilogue => false;

    /// <summary>Optional fused op: Gemm with a LoRA expand added in the epilogue.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">The backend does not provide the fused form.</exception>
    static virtual void GemmLoraEpilogue(in GemmLoraEpilogueArgs a) => throw new NotSupportedException(nameof(GemmLoraEpilogue));

    /// <summary>Reserved, M1: Gated delta rule, chunked (prefill) or recurrent (decode), with indexed recurrent-state slots (S3) and row-strided q, k and v views.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void GatedDeltaRule(in GatedDeltaRuleArgs a) => throw new NotSupportedException(nameof(GatedDeltaRule));

    /// <summary>Reserved, M1: Depthwise causal 1-D convolution with a rolling conv state per sequence.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void CausalConv1d(in CausalConv1dArgs a) => throw new NotSupportedException(nameof(CausalConv1d));

    /// <summary>Reserved, M1: Copy recurrent-state slots (state-group fork).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void RecurrentStateCopy(in RecurrentStateCopyArgs a) => throw new NotSupportedException(nameof(RecurrentStateCopy));

    /// <summary>Reserved, M1: Write recurrent-state slots to the snapshot pool, or restore them.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void RecurrentStateSnapshot(in RecurrentStateSnapshotArgs a) => throw new NotSupportedException(nameof(RecurrentStateSnapshot));

    /// <summary>Reserved, M1: Paged attention restricted to a sliding window.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void SlidingWindowAttention(in SlidingWindowAttentionArgs a) => throw new NotSupportedException(nameof(SlidingWindowAttention));

    /// <summary>
    /// Qwen3.5 gating (contract version 3): Gated DeltaNet gates from the raw a and b projections,
    /// <c>G = -exp(ALog) * softplus(A + DtBias)</c> and <c>Beta = sigmoid(B)</c>, per token and value head.
    /// </summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void GatedDeltaRuleGates(in GatedDeltaRuleGatesArgs a) => throw new NotSupportedException(nameof(GatedDeltaRuleGates));

    /// <summary>
    /// Qwen3.5 gating (contract version 3): RMS norm before a SiLU gate,
    /// <c>Output = Input * rsqrt(mean(Input^2) + Eps) * Weight * silu(Gate)</c>, per row.
    /// </summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void GatedRmsNorm(in GatedRmsNormArgs a) => throw new NotSupportedException(nameof(GatedRmsNorm));

    /// <summary>Qwen3.5 gating (contract version 3): the attention output gate, <c>Output = Input * sigmoid(Gate)</c>.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void SigmoidMul(in SigmoidMulArgs a) => throw new NotSupportedException(nameof(SigmoidMul));

    /// <summary>Reserved, M2: Rotary embedding with YaRN, longrope or dynamic NTK scaling.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void RopeScaled(in RopeScaledArgs a) => throw new NotSupportedException(nameof(RopeScaled));

    /// <summary>Reserved, M2: Quantise rows to an 8-bit or 4-bit type with per-row or per-group scales.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void Quantize(in QuantizeArgs a) => throw new NotSupportedException(nameof(Quantize));

    /// <summary>Reserved, M2: Dequantise rows with their scales.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void Dequantize(in DequantizeArgs a) => throw new NotSupportedException(nameof(Dequantize));

    /// <summary>Reserved, M2: Gemm with 4-bit weights and 16-bit activations (Marlin layout, S6).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void GemmW4A16(in GemmW4A16Args a) => throw new NotSupportedException(nameof(GemmW4A16));

    /// <summary>Reserved, M2: Gemm with FP8 inputs and per-tensor or per-channel scales (SM 89 and later).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void GemmFp8(in GemmFp8Args a) => throw new NotSupportedException(nameof(GemmFp8));

    /// <summary>Reserved, M2: Attention for a speculation tree with a per-sequence tree mask.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void TreeAttention(in TreeAttentionArgs a) => throw new NotSupportedException(nameof(TreeAttention));

    /// <summary>Reserved, M2: Speculative rejection sampling of draft tokens against target probabilities.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void RejectSample(in RejectSampleArgs a) => throw new NotSupportedException(nameof(RejectSample));

    /// <summary>Reserved, M2: Router softmax, top-k and optional renormalisation (vLLM top-k kernel, S6).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void MoeRouteTopK(in MoeRouteTopKArgs a) => throw new NotSupportedException(nameof(MoeRouteTopK));

    /// <summary>Reserved, M2: Group token rows by expert; per-expert counts and offsets are written on the device.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void MoePermute(in MoePermuteArgs a) => throw new NotSupportedException(nameof(MoePermute));

    /// <summary>Reserved, M2: Weighted combine of expert outputs back to token order.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void MoeUnpermute(in MoeUnpermuteArgs a) => throw new NotSupportedException(nameof(MoeUnpermute));

    /// <summary>Reserved, M2: One Gemm per expert; problem sizes are read from device memory so the call stays graph-capturable (S6).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void GroupedGemm(in GroupedGemmArgs a) => throw new NotSupportedException(nameof(GroupedGemm));

    /// <summary>Reserved, M2: Route, permute, two grouped Gemms with SiluAndMul, and combine, as one op.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void FusedMoe(in FusedMoeArgs a) => throw new NotSupportedException(nameof(FusedMoe));

    /// <summary>Reserved, M3: Batched LoRA A projection with an adapter per row.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void LoraShrink(in LoraShrinkArgs a) => throw new NotSupportedException(nameof(LoraShrink));

    /// <summary>Reserved, M3: Batched LoRA B projection added to the output.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void LoraExpand(in LoraExpandArgs a) => throw new NotSupportedException(nameof(LoraExpand));

    /// <summary>Reserved, M4: Sum across the tensor-parallel group, in place (C9).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void AllReduce(in AllReduceArgs a) => throw new NotSupportedException(nameof(AllReduce));

    /// <summary>Reserved, M4: Gather shards from every rank of a group.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void AllGather(in AllGatherArgs a) => throw new NotSupportedException(nameof(AllGather));

    /// <summary>Reserved, M4: Sum then scatter shards.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void ReduceScatter(in ReduceScatterArgs a) => throw new NotSupportedException(nameof(ReduceScatter));

    /// <summary>Reserved, M4: Exchange expert rows between ranks (expert parallelism).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void AllToAll(in AllToAllArgs a) => throw new NotSupportedException(nameof(AllToAll));

    /// <summary>Reserved, M4: Multi-head latent attention over a latent KV pool (C4 MLA latent groups).</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void MlaAttention(in MlaAttentionArgs a) => throw new NotSupportedException(nameof(MlaAttention));

    /// <summary>Reserved, M5: Image or video patches to embeddings.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void PatchEmbed(in PatchEmbedArgs a) => throw new NotSupportedException(nameof(PatchEmbed));

    /// <summary>Reserved, M5: Unmasked attention for encoders over contiguous (not paged) K and V.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void BidirectionalAttention(in BidirectionalAttentionArgs a) => throw new NotSupportedException(nameof(BidirectionalAttention));

    /// <summary>Reserved, M5: Log-mel spectrogram of audio samples.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void MelSpectrogram(in MelSpectrogramArgs a) => throw new NotSupportedException(nameof(MelSpectrogram));

    /// <summary>Reserved, M5: Convolutional subsampling of spectrogram frames.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void AudioConvSubsample(in AudioConvSubsampleArgs a) => throw new NotSupportedException(nameof(AudioConvSubsample));

    /// <summary>Reserved, M5: Dequantise GGUF block-quantised weights.</summary>
    /// <param name="a">Arguments.</param>
    /// <exception cref="NotSupportedException">Until the backend implements it.</exception>
    static virtual void DequantizeGguf(in DequantizeGgufArgs a) => throw new NotSupportedException(nameof(DequantizeGguf));

}
