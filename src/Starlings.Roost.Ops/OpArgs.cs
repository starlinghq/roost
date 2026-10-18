using System.Runtime.InteropServices;
using Starlings.Roost.Abi;

namespace Starlings.Roost.Ops;

// Argument structs of the C3 op vocabulary. Buffers are DeviceSpan (address, length, element type); there is no tensor
// type. Device addresses are worker-side only and never appear in C2 or shared memory. Field names are what the
// source generator fills (S4 finding 5). Reserved ops' arguments are a v0 draft; an item that lands one may change
// its arguments with a contract version bump, since no backend implements it yet.


/// <summary>Arguments of <see cref="IRoostOps.RmsNorm"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RmsNormArgs
{
    /// <summary>[Rows, Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Dim].</summary>
    public DeviceSpan Weight;

    /// <summary>[Rows, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Row width.</summary>
    public int Dim;

    /// <summary>Epsilon inside the square root.</summary>
    public float Eps;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.QkNorm"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct QkNormArgs
{
    /// <summary>[Tokens, QHeads, HeadDim].</summary>
    public DeviceSpan Q;

    /// <summary>[Tokens, KvHeads, HeadDim].</summary>
    public DeviceSpan K;

    /// <summary>[HeadDim].</summary>
    public DeviceSpan QWeight;

    /// <summary>[HeadDim].</summary>
    public DeviceSpan KWeight;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Query heads.</summary>
    public int QHeads;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Epsilon.</summary>
    public float Eps;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.Rope"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RopeArgs
{
    /// <summary>[Tokens, QHeads, HeadDim].</summary>
    public DeviceSpan Q;

    /// <summary>[Tokens, KvHeads, HeadDim].</summary>
    public DeviceSpan K;

    /// <summary>[Tokens] int32.</summary>
    public DeviceSpan Positions;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Query heads.</summary>
    public int QHeads;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Rotated dimensions; 0 means HeadDim.</summary>
    public int RotaryDim;

    /// <summary>Default or llama3.</summary>
    public RopeKind Kind;

    /// <summary>GPT-J style pairs instead of NeoX halves.</summary>
    public bool Interleaved;

    /// <summary>Base frequency.</summary>
    public float Theta;

    /// <summary>Llama 3 scaling factor.</summary>
    public float Llama3Factor;

    /// <summary>Llama 3 low-frequency factor.</summary>
    public float Llama3LowFreqFactor;

    /// <summary>Llama 3 high-frequency factor.</summary>
    public float Llama3HighFreqFactor;

    /// <summary>Llama 3 original context length.</summary>
    public int Llama3OriginalMaxPosition;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.PagedAttention"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct PagedAttentionArgs
{
    /// <summary>[Tokens, QHeads, HeadDim].</summary>
    public DeviceSpan Q;

    /// <summary>[Tokens, QHeads, HeadDim].</summary>
    public DeviceSpan Output;

    /// <summary>Key pool.</summary>
    public DeviceSpan KCache;

    /// <summary>Value pool.</summary>
    public DeviceSpan VCache;

    /// <summary>[Batch + 1] int32 token offsets per sequence.</summary>
    public DeviceSpan QIndptr;

    /// <summary>[Batch + 1] int32 offsets into KvIndices.</summary>
    public DeviceSpan KvIndptr;

    /// <summary>Block ids.</summary>
    public DeviceSpan KvIndices;

    /// <summary>[Batch] int32.</summary>
    public DeviceSpan KvLastPageLength;

    /// <summary>Decode or prefill.</summary>
    public AttentionMode Mode;

    /// <summary>Sequences.</summary>
    public int Batch;

    /// <summary>Query tokens.</summary>
    public int Tokens;

    /// <summary>Query heads.</summary>
    public int QHeads;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Tokens per block.</summary>
    public int BlockSize;

    /// <summary>Pool layout.</summary>
    public TensorLayout KvLayout;

    /// <summary>Softmax scale; 0 means 1/sqrt(HeadDim).</summary>
    public float Scale;

    /// <summary>Logit soft cap; 0 disables.</summary>
    public float SoftCap;

    /// <summary>Sliding window; -1 disables.</summary>
    public int WindowLeft;

    /// <summary>Causal mask (prefill).</summary>
    public bool Causal;

    /// <summary>KV cache element type.</summary>
    public DType KvDType;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.SiluAndMul"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SiluAndMulArgs
{
    /// <summary>[Rows, 2 * Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Rows, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Output width.</summary>
    public int Dim;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.GeluAndMul"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GeluAndMulArgs
{
    /// <summary>[Rows, 2 * Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Rows, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Output width.</summary>
    public int Dim;

    /// <summary>Exact or tanh.</summary>
    public GeluApproximation Approximation;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.Gemm"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GemmArgs
{
    /// <summary>[M, K].</summary>
    public DeviceSpan A;

    /// <summary>[K, N], or [N, K] when TransposeB.</summary>
    public DeviceSpan B;

    /// <summary>Optional addend.</summary>
    public DeviceSpan C;

    /// <summary>[M, N] output.</summary>
    public DeviceSpan D;

    /// <summary>Optional [N].</summary>
    public DeviceSpan Bias;

    /// <summary>Rows.</summary>
    public long M;

    /// <summary>Columns.</summary>
    public long N;

    /// <summary>Inner dimension.</summary>
    public long K;

    /// <summary>B is [N, K] (weights as stored).</summary>
    public bool TransposeB;

    /// <summary>Scale of the product.</summary>
    public float Alpha;

    /// <summary>Scale of C.</summary>
    public float Beta;

    /// <summary>Shape bucket of the step; selects the pinned algorithm.</summary>
    public int ShapeBucket;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>Weight element type.</summary>
    public DType WeightDType;
}

/// <summary>Arguments of <see cref="IRoostOps.Embedding"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct EmbeddingArgs
{
    /// <summary>[Tokens] int32.</summary>
    public DeviceSpan Ids;

    /// <summary>[Vocab, Dim].</summary>
    public DeviceSpan Table;

    /// <summary>[Tokens, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Width.</summary>
    public int Dim;

    /// <summary>Rows of the table.</summary>
    public int Vocab;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.LmHead"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct LmHeadArgs
{
    /// <summary>[Tokens, Dim].</summary>
    public DeviceSpan Hidden;

    /// <summary>[Selected] int32 indices into Hidden.</summary>
    public DeviceSpan Rows;

    /// <summary>[Vocab, Dim].</summary>
    public DeviceSpan Weight;

    /// <summary>[Selected, VocabPadded] float32.</summary>
    public DeviceSpan Logits;

    /// <summary>Selected rows.</summary>
    public int Selected;

    /// <summary>Hidden width.</summary>
    public int Dim;

    /// <summary>Vocabulary size.</summary>
    public int Vocab;

    /// <summary>Logit row stride.</summary>
    public int VocabPadded;

    /// <summary>Final logit soft cap; 0 disables.</summary>
    public float SoftCap;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.Residual"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct ResidualArgs
{
    /// <summary>First addend.</summary>
    public DeviceSpan A;

    /// <summary>Second addend.</summary>
    public DeviceSpan B;

    /// <summary>Output.</summary>
    public DeviceSpan Output;

    /// <summary>Element count.</summary>
    public long Elements;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.CacheWrite"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct CacheWriteArgs
{
    /// <summary>[Tokens, KvHeads, HeadDim].</summary>
    public DeviceSpan K;

    /// <summary>[Tokens, KvHeads, HeadDim].</summary>
    public DeviceSpan V;

    /// <summary>Key pool.</summary>
    public DeviceSpan KCache;

    /// <summary>Value pool.</summary>
    public DeviceSpan VCache;

    /// <summary>[Tokens] int64 block * BlockSize + offset; negative skips.</summary>
    public DeviceSpan SlotMapping;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Tokens per block.</summary>
    public int BlockSize;

    /// <summary>Pool layout.</summary>
    public TensorLayout KvLayout;

    /// <summary>Key scale for quantised KV; 1 otherwise.</summary>
    public float KScale;

    /// <summary>Value scale for quantised KV; 1 otherwise.</summary>
    public float VScale;

    /// <summary>KV cache element type.</summary>
    public DType KvDType;
}

/// <summary>Arguments of <see cref="IRoostOps.CacheCopy"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct CacheCopyArgs
{
    /// <summary>Source pool.</summary>
    public DeviceSpan Source;

    /// <summary>Destination pool; may equal Source.</summary>
    public DeviceSpan Destination;

    /// <summary>[Count, 2] int32 (source block, destination block).</summary>
    public DeviceSpan Pairs;

    /// <summary>Bytes per block.</summary>
    public long BlockBytes;

    /// <summary>Pairs.</summary>
    public int Count;
}

/// <summary>Arguments of <see cref="IRoostOps.Argmax"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct ArgmaxArgs
{
    /// <summary>[Rows, Stride] float32.</summary>
    public DeviceSpan Logits;

    /// <summary>[Rows] int32.</summary>
    public DeviceSpan Ids;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Vocabulary size.</summary>
    public int Vocab;

    /// <summary>Row stride; 0 means Vocab.</summary>
    public int Stride;
}

/// <summary>Arguments of <see cref="IRoostOps.Sample"/>. M0.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SampleArgs
{
    /// <summary>[Rows, Stride] float32.</summary>
    public DeviceSpan Logits;

    /// <summary>[Rows] float32; 0 means greedy.</summary>
    public DeviceSpan Temperature;

    /// <summary>[Rows] int32; 0 disables.</summary>
    public DeviceSpan TopK;

    /// <summary>[Rows] float32; 1 disables.</summary>
    public DeviceSpan TopP;

    /// <summary>[Rows] float32; 0 disables.</summary>
    public DeviceSpan MinP;

    /// <summary>[Rows] uint64.</summary>
    public DeviceSpan Seeds;

    /// <summary>[Rows] uint64 Philox offsets.</summary>
    public DeviceSpan Offsets;

    /// <summary>Optional [Rows, ceil(Vocab / 32)] uint32 allowed-token bitmask.</summary>
    public DeviceSpan Mask;

    /// <summary>[Rows] int32 output.</summary>
    public DeviceSpan Ids;

    /// <summary>Optional [Rows] float32 output.</summary>
    public DeviceSpan Logprobs;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Vocabulary size.</summary>
    public int Vocab;

    /// <summary>Row stride.</summary>
    public int Stride;
}

/// <summary>Arguments of <see cref="IRoostOps.AddRmsNorm"/>. Optional fused op.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct AddRmsNormArgs
{
    /// <summary>[Rows, Dim], overwritten with the normalised result.</summary>
    public DeviceSpan Input;

    /// <summary>[Rows, Dim], overwritten with the sum.</summary>
    public DeviceSpan Residual;

    /// <summary>[Dim].</summary>
    public DeviceSpan Weight;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Row width.</summary>
    public int Dim;

    /// <summary>Epsilon.</summary>
    public float Eps;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.AddRmsNormQuant"/>. Optional fused op.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct AddRmsNormQuantArgs
{
    /// <summary>[Rows, Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Rows, Dim].</summary>
    public DeviceSpan Residual;

    /// <summary>[Dim].</summary>
    public DeviceSpan Weight;

    /// <summary>[Rows, Dim] quantised.</summary>
    public DeviceSpan Output;

    /// <summary>Output scales.</summary>
    public DeviceSpan Scales;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Row width.</summary>
    public int Dim;

    /// <summary>Epsilon.</summary>
    public float Eps;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>Quantised type.</summary>
    public DType OutputDType;
}

/// <summary>Arguments of <see cref="IRoostOps.SiluMulQuant"/>. Optional fused op.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SiluMulQuantArgs
{
    /// <summary>[Rows, 2 * Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Rows, Dim] quantised.</summary>
    public DeviceSpan Output;

    /// <summary>Output scales.</summary>
    public DeviceSpan Scales;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Output width.</summary>
    public int Dim;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>Quantised type.</summary>
    public DType OutputDType;
}

/// <summary>Arguments of <see cref="IRoostOps.QkvQkNormRopeCacheWrite"/>. Optional fused op.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct QkvQkNormRopeCacheWriteArgs
{
    /// <summary>[Tokens, (QHeads + 2 * KvHeads) * HeadDim].</summary>
    public DeviceSpan Qkv;

    /// <summary>[Tokens, QHeads, HeadDim] output.</summary>
    public DeviceSpan Q;

    /// <summary>QkNorm query weight.</summary>
    public DeviceSpan QWeight;

    /// <summary>QkNorm key weight.</summary>
    public DeviceSpan KWeight;

    /// <summary>[Tokens] int32.</summary>
    public DeviceSpan Positions;

    /// <summary>Key pool.</summary>
    public DeviceSpan KCache;

    /// <summary>Value pool.</summary>
    public DeviceSpan VCache;

    /// <summary>[Tokens] int64.</summary>
    public DeviceSpan SlotMapping;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Query heads.</summary>
    public int QHeads;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Tokens per block.</summary>
    public int BlockSize;

    /// <summary>QkNorm epsilon.</summary>
    public float Eps;

    /// <summary>Rope base frequency.</summary>
    public float Theta;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>KV cache element type.</summary>
    public DType KvDType;
}

/// <summary>Arguments of <see cref="IRoostOps.GemmLoraEpilogue"/>. Optional fused op.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GemmLoraEpilogueArgs
{
    /// <summary>[M, K].</summary>
    public DeviceSpan A;

    /// <summary>[N, K] weights.</summary>
    public DeviceSpan B;

    /// <summary>[M, N].</summary>
    public DeviceSpan D;

    /// <summary>[M, Rank] output of LoraShrink.</summary>
    public DeviceSpan LoraShrunk;

    /// <summary>Stacked [Adapters, N, Rank].</summary>
    public DeviceSpan LoraB;

    /// <summary>[M] int32 adapter per row, -1 for none.</summary>
    public DeviceSpan LoraIndices;

    /// <summary>Rows.</summary>
    public long M;

    /// <summary>Columns.</summary>
    public long N;

    /// <summary>Inner dimension.</summary>
    public long K;

    /// <summary>LoRA rank.</summary>
    public int Rank;

    /// <summary>LoRA scaling.</summary>
    public float LoraScale;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.GatedDeltaRule"/>. Reserved, M1.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GatedDeltaRuleArgs
{
    /// <summary>[Tokens, KHeads, KDim].</summary>
    public DeviceSpan Q;

    /// <summary>[Tokens, KHeads, KDim].</summary>
    public DeviceSpan K;

    /// <summary>[Tokens, VHeads, VDim].</summary>
    public DeviceSpan V;

    /// <summary>[Tokens, VHeads] log decay.</summary>
    public DeviceSpan G;

    /// <summary>[Tokens, VHeads].</summary>
    public DeviceSpan Beta;

    /// <summary>[Tokens, VHeads, VDim].</summary>
    public DeviceSpan Output;

    /// <summary>Recurrent-state pool [Slots, VHeads, KDim, VDim] float32.</summary>
    public DeviceSpan State;

    /// <summary>[Batch] int32 slot per sequence (C4 recurrent pool).</summary>
    public DeviceSpan StateSlots;

    /// <summary>[Batch + 1] int32.</summary>
    public DeviceSpan CuSeqlens;

    /// <summary>Sequences.</summary>
    public int Batch;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Key heads.</summary>
    public int KHeads;

    /// <summary>Value heads.</summary>
    public int VHeads;

    /// <summary>Key head size.</summary>
    public int KDim;

    /// <summary>Value head size.</summary>
    public int VDim;

    /// <summary>Query scale.</summary>
    public float Scale;

    /// <summary>Read the slot's state before the first token.</summary>
    public bool UseInitialState;

    /// <summary>Normalise q and k inside the op.</summary>
    public bool L2NormQk;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.CausalConv1d"/>. Reserved, M1.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct CausalConv1dArgs
{
    /// <summary>[Tokens, Channels].</summary>
    public DeviceSpan Input;

    /// <summary>[Channels, Width].</summary>
    public DeviceSpan Weight;

    /// <summary>Optional [Channels].</summary>
    public DeviceSpan Bias;

    /// <summary>[Tokens, Channels].</summary>
    public DeviceSpan Output;

    /// <summary>Pool [Slots, Channels, Width - 1].</summary>
    public DeviceSpan ConvState;

    /// <summary>[Batch] int32.</summary>
    public DeviceSpan StateSlots;

    /// <summary>[Batch + 1] int32.</summary>
    public DeviceSpan CuSeqlens;

    /// <summary>Sequences.</summary>
    public int Batch;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Channels.</summary>
    public int Channels;

    /// <summary>Kernel width.</summary>
    public int Width;

    /// <summary>Apply SiLU.</summary>
    public bool Silu;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.RecurrentStateCopy"/>. Reserved, M1.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RecurrentStateCopyArgs
{
    /// <summary>State pool.</summary>
    public DeviceSpan Pool;

    /// <summary>[Count, 2] int32 (source slot, destination slot).</summary>
    public DeviceSpan Pairs;

    /// <summary>Bytes per slot.</summary>
    public long SlotBytes;

    /// <summary>Pairs.</summary>
    public int Count;
}

/// <summary>Arguments of <see cref="IRoostOps.RecurrentStateSnapshot"/>. Reserved, M1.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RecurrentStateSnapshotArgs
{
    /// <summary>State pool.</summary>
    public DeviceSpan Pool;

    /// <summary>Snapshot pool (C4: its own pool).</summary>
    public DeviceSpan Snapshots;

    /// <summary>[Count, 2] int32 (slot, snapshot).</summary>
    public DeviceSpan Pairs;

    /// <summary>Bytes per slot.</summary>
    public long SlotBytes;

    /// <summary>Pairs.</summary>
    public int Count;

    /// <summary>Snapshot to slot instead of slot to snapshot.</summary>
    public bool Restore;
}

/// <summary>Arguments of <see cref="IRoostOps.SlidingWindowAttention"/>. Reserved, M1.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct SlidingWindowAttentionArgs
{
    /// <summary>Paged attention arguments; WindowLeft is the window.</summary>
    public PagedAttentionArgs Attention;

    /// <summary>Window in tokens.</summary>
    public int Window;
}

/// <summary>Arguments of <see cref="IRoostOps.RopeScaled"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RopeScaledArgs
{
    /// <summary>Query.</summary>
    public DeviceSpan Q;

    /// <summary>Key.</summary>
    public DeviceSpan K;

    /// <summary>[Tokens] int32.</summary>
    public DeviceSpan Positions;

    /// <summary>Precomputed [MaxPosition, RotaryDim] cos and sin.</summary>
    public DeviceSpan CosSinCache;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Query heads.</summary>
    public int QHeads;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Rotated dimensions.</summary>
    public int RotaryDim;

    /// <summary>1 YaRN, 2 longrope, 3 dynamic NTK.</summary>
    public int ScalingKind;

    /// <summary>Scaling factor.</summary>
    public float Factor;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.Quantize"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct QuantizeArgs
{
    /// <summary>Input.</summary>
    public DeviceSpan Input;

    /// <summary>Output.</summary>
    public DeviceSpan Output;

    /// <summary>Scales.</summary>
    public DeviceSpan Scales;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Row width.</summary>
    public int Dim;

    /// <summary>Elements per scale; 0 means per row.</summary>
    public int GroupSize;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>Quantised type.</summary>
    public DType OutputDType;
}

/// <summary>Arguments of <see cref="IRoostOps.Dequantize"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct DequantizeArgs
{
    /// <summary>Input.</summary>
    public DeviceSpan Input;

    /// <summary>Scales.</summary>
    public DeviceSpan Scales;

    /// <summary>Optional zero points.</summary>
    public DeviceSpan Zeros;

    /// <summary>Output.</summary>
    public DeviceSpan Output;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Row width.</summary>
    public int Dim;

    /// <summary>Elements per scale.</summary>
    public int GroupSize;

    /// <summary>Quantised type.</summary>
    public DType InputDType;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.GemmW4A16"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GemmW4A16Args
{
    /// <summary>[M, K].</summary>
    public DeviceSpan A;

    /// <summary>Packed 4-bit [N, K] in the kernel's repacked layout.</summary>
    public DeviceSpan B;

    /// <summary>Group scales.</summary>
    public DeviceSpan Scales;

    /// <summary>Optional zero points.</summary>
    public DeviceSpan Zeros;

    /// <summary>[M, N].</summary>
    public DeviceSpan D;

    /// <summary>Rows.</summary>
    public long M;

    /// <summary>Columns.</summary>
    public long N;

    /// <summary>Inner dimension.</summary>
    public long K;

    /// <summary>Weights per scale.</summary>
    public int GroupSize;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.GemmFp8"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GemmFp8Args
{
    /// <summary>[M, K] FP8.</summary>
    public DeviceSpan A;

    /// <summary>[N, K] FP8.</summary>
    public DeviceSpan B;

    /// <summary>Activation scales.</summary>
    public DeviceSpan AScale;

    /// <summary>Weight scales.</summary>
    public DeviceSpan BScale;

    /// <summary>[M, N].</summary>
    public DeviceSpan D;

    /// <summary>Rows.</summary>
    public long M;

    /// <summary>Columns.</summary>
    public long N;

    /// <summary>Inner dimension.</summary>
    public long K;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.TreeAttention"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct TreeAttentionArgs
{
    /// <summary>Paged attention arguments.</summary>
    public PagedAttentionArgs Attention;

    /// <summary>Per-node uint64 mask words.</summary>
    public DeviceSpan TreeMask;

    /// <summary>Per-node int32 parent, -1 for the root.</summary>
    public DeviceSpan TreeParents;
}

/// <summary>Arguments of <see cref="IRoostOps.RejectSample"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct RejectSampleArgs
{
    /// <summary>[Rows, Vocab].</summary>
    public DeviceSpan TargetLogits;

    /// <summary>Optional [Rows, Vocab]; null for greedy drafts.</summary>
    public DeviceSpan DraftProbs;

    /// <summary>[Rows] int32.</summary>
    public DeviceSpan DraftIds;

    /// <summary>[Sequences] uint64.</summary>
    public DeviceSpan Seeds;

    /// <summary>[Sequences] int32 accepted count.</summary>
    public DeviceSpan Accepted;

    /// <summary>[Rows + Sequences] int32.</summary>
    public DeviceSpan OutputIds;

    /// <summary>Sequences.</summary>
    public int Sequences;

    /// <summary>Draft rows.</summary>
    public int Rows;

    /// <summary>Vocabulary size.</summary>
    public int Vocab;
}

/// <summary>Arguments of <see cref="IRoostOps.MoeRouteTopK"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct MoeRouteTopKArgs
{
    /// <summary>[Tokens, Experts].</summary>
    public DeviceSpan Logits;

    /// <summary>[Tokens, TopK] float32.</summary>
    public DeviceSpan Weights;

    /// <summary>[Tokens, TopK] int32.</summary>
    public DeviceSpan ExpertIds;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Experts.</summary>
    public int Experts;

    /// <summary>Experts per token.</summary>
    public int TopK;

    /// <summary>Renormalise the top-k weights.</summary>
    public bool Renormalize;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.MoePermute"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct MoePermuteArgs
{
    /// <summary>[Tokens, Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Tokens, TopK] int32.</summary>
    public DeviceSpan ExpertIds;

    /// <summary>[Tokens * TopK, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>[Experts] int32.</summary>
    public DeviceSpan ExpertCounts;

    /// <summary>[Experts + 1] int32.</summary>
    public DeviceSpan ExpertOffsets;

    /// <summary>[Tokens * TopK] int32.</summary>
    public DeviceSpan Permutation;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Width.</summary>
    public int Dim;

    /// <summary>Experts.</summary>
    public int Experts;

    /// <summary>Experts per token.</summary>
    public int TopK;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.MoeUnpermute"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct MoeUnpermuteArgs
{
    /// <summary>[Tokens * TopK, Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Tokens * TopK] int32.</summary>
    public DeviceSpan Permutation;

    /// <summary>[Tokens, TopK].</summary>
    public DeviceSpan Weights;

    /// <summary>[Tokens, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Width.</summary>
    public int Dim;

    /// <summary>Experts per token.</summary>
    public int TopK;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.GroupedGemm"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct GroupedGemmArgs
{
    /// <summary>Permuted rows [Rows, K].</summary>
    public DeviceSpan A;

    /// <summary>Stacked weights [Experts, N, K].</summary>
    public DeviceSpan B;

    /// <summary>[Rows, N].</summary>
    public DeviceSpan D;

    /// <summary>[Experts + 1] int32 row offsets on the device.</summary>
    public DeviceSpan ExpertOffsets;

    /// <summary>Experts (groups).</summary>
    public int Experts;

    /// <summary>Columns.</summary>
    public long N;

    /// <summary>Inner dimension.</summary>
    public long K;

    /// <summary>Upper bound on rows, for workspace sizing.</summary>
    public long MaxRows;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>Weight element type.</summary>
    public DType WeightDType;
}

/// <summary>Arguments of <see cref="IRoostOps.FusedMoe"/>. Reserved, M2.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct FusedMoeArgs
{
    /// <summary>[Tokens, Dim].</summary>
    public DeviceSpan Input;

    /// <summary>[Tokens, Experts].</summary>
    public DeviceSpan RouterLogits;

    /// <summary>[Experts, 2 * Intermediate, Dim].</summary>
    public DeviceSpan W13;

    /// <summary>[Experts, Dim, Intermediate].</summary>
    public DeviceSpan W2;

    /// <summary>[Tokens, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Tokens.</summary>
    public int Tokens;

    /// <summary>Hidden width.</summary>
    public int Dim;

    /// <summary>Expert intermediate width.</summary>
    public int Intermediate;

    /// <summary>Experts.</summary>
    public int Experts;

    /// <summary>Experts per token.</summary>
    public int TopK;

    /// <summary>Renormalise the top-k weights.</summary>
    public bool Renormalize;

    /// <summary>Activation element type.</summary>
    public DType DType;

    /// <summary>Weight element type.</summary>
    public DType WeightDType;
}

/// <summary>Arguments of <see cref="IRoostOps.LoraShrink"/>. Reserved, M3.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct LoraShrinkArgs
{
    /// <summary>[Rows, K].</summary>
    public DeviceSpan Input;

    /// <summary>Stacked [Adapters, Rank, K].</summary>
    public DeviceSpan LoraA;

    /// <summary>[Rows] int32 adapter, -1 for none.</summary>
    public DeviceSpan Indices;

    /// <summary>[Rows, Rank].</summary>
    public DeviceSpan Output;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Input width.</summary>
    public int K;

    /// <summary>LoRA rank.</summary>
    public int Rank;

    /// <summary>Scaling.</summary>
    public float Scale;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.LoraExpand"/>. Reserved, M3.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct LoraExpandArgs
{
    /// <summary>[Rows, Rank].</summary>
    public DeviceSpan Input;

    /// <summary>Stacked [Adapters, N, Rank].</summary>
    public DeviceSpan LoraB;

    /// <summary>[Rows] int32.</summary>
    public DeviceSpan Indices;

    /// <summary>[Rows, N], accumulated.</summary>
    public DeviceSpan Output;

    /// <summary>Rows (tokens).</summary>
    public int Rows;

    /// <summary>Output width.</summary>
    public int N;

    /// <summary>LoRA rank.</summary>
    public int Rank;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.AllReduce"/>. Reserved, M4.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct AllReduceArgs
{
    /// <summary>Data.</summary>
    public DeviceSpan Buffer;

    /// <summary>Elements.</summary>
    public long Elements;

    /// <summary>Communicator group from the mesh.</summary>
    public int Group;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.AllGather"/>. Reserved, M4.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct AllGatherArgs
{
    /// <summary>Local shard.</summary>
    public DeviceSpan Input;

    /// <summary>Gathered.</summary>
    public DeviceSpan Output;

    /// <summary>Elements per rank.</summary>
    public long Elements;

    /// <summary>Communicator group.</summary>
    public int Group;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.ReduceScatter"/>. Reserved, M4.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct ReduceScatterArgs
{
    /// <summary>Full buffer.</summary>
    public DeviceSpan Input;

    /// <summary>Local shard.</summary>
    public DeviceSpan Output;

    /// <summary>Elements per rank.</summary>
    public long Elements;

    /// <summary>Communicator group.</summary>
    public int Group;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.AllToAll"/>. Reserved, M4.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct AllToAllArgs
{
    /// <summary>Send rows.</summary>
    public DeviceSpan Input;

    /// <summary>Receive rows.</summary>
    public DeviceSpan Output;

    /// <summary>[Ranks] int32.</summary>
    public DeviceSpan SendCounts;

    /// <summary>[Ranks] int32.</summary>
    public DeviceSpan ReceiveCounts;

    /// <summary>Row width.</summary>
    public int Dim;

    /// <summary>Communicator group.</summary>
    public int Group;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.MlaAttention"/>. Reserved, M4.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct MlaAttentionArgs
{
    /// <summary>[Tokens, Heads, QkDim].</summary>
    public DeviceSpan Q;

    /// <summary>Output.</summary>
    public DeviceSpan Output;

    /// <summary>Latent pool.</summary>
    public DeviceSpan LatentCache;

    /// <summary>[Batch + 1] int32.</summary>
    public DeviceSpan KvIndptr;

    /// <summary>Block ids.</summary>
    public DeviceSpan KvIndices;

    /// <summary>Sequences.</summary>
    public int Batch;

    /// <summary>Heads.</summary>
    public int Heads;

    /// <summary>Latent width.</summary>
    public int LatentDim;

    /// <summary>Decoupled rope width.</summary>
    public int RopeDim;

    /// <summary>Tokens per block.</summary>
    public int BlockSize;

    /// <summary>Softmax scale.</summary>
    public float Scale;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.PatchEmbed"/>. Reserved, M5.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct PatchEmbedArgs
{
    /// <summary>Pixels.</summary>
    public DeviceSpan Input;

    /// <summary>Patch projection.</summary>
    public DeviceSpan Weight;

    /// <summary>Optional bias.</summary>
    public DeviceSpan Bias;

    /// <summary>[Patches, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Patches.</summary>
    public int Patches;

    /// <summary>Patch edge in pixels.</summary>
    public int PatchSize;

    /// <summary>Input channels.</summary>
    public int Channels;

    /// <summary>Embedding width.</summary>
    public int Dim;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.BidirectionalAttention"/>. Reserved, M5.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct BidirectionalAttentionArgs
{
    /// <summary>[Tokens, Heads, HeadDim].</summary>
    public DeviceSpan Q;

    /// <summary>[Tokens, KvHeads, HeadDim].</summary>
    public DeviceSpan K;

    /// <summary>[Tokens, KvHeads, HeadDim].</summary>
    public DeviceSpan V;

    /// <summary>Output.</summary>
    public DeviceSpan Output;

    /// <summary>[Batch + 1] int32.</summary>
    public DeviceSpan CuSeqlens;

    /// <summary>Sequences.</summary>
    public int Batch;

    /// <summary>Heads.</summary>
    public int Heads;

    /// <summary>KV heads.</summary>
    public int KvHeads;

    /// <summary>Head size.</summary>
    public int HeadDim;

    /// <summary>Softmax scale.</summary>
    public float Scale;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.MelSpectrogram"/>. Reserved, M5.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct MelSpectrogramArgs
{
    /// <summary>Samples, float32.</summary>
    public DeviceSpan Input;

    /// <summary>[Frames, Mels].</summary>
    public DeviceSpan Output;

    /// <summary>Samples.</summary>
    public long Samples;

    /// <summary>Hz.</summary>
    public int SampleRate;

    /// <summary>FFT size.</summary>
    public int FftSize;

    /// <summary>Hop.</summary>
    public int HopLength;

    /// <summary>Mel bins.</summary>
    public int Mels;
}

/// <summary>Arguments of <see cref="IRoostOps.AudioConvSubsample"/>. Reserved, M5.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct AudioConvSubsampleArgs
{
    /// <summary>[Frames, Mels].</summary>
    public DeviceSpan Input;

    /// <summary>Convolution weights.</summary>
    public DeviceSpan Weight;

    /// <summary>Bias.</summary>
    public DeviceSpan Bias;

    /// <summary>[Frames / Factor, Dim].</summary>
    public DeviceSpan Output;

    /// <summary>Frames.</summary>
    public int Frames;

    /// <summary>Subsampling factor.</summary>
    public int Factor;

    /// <summary>Output width.</summary>
    public int Dim;

    /// <summary>Activation element type.</summary>
    public DType DType;
}

/// <summary>Arguments of <see cref="IRoostOps.DequantizeGguf"/>. Reserved, M5.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct DequantizeGgufArgs
{
    /// <summary>GGUF blocks.</summary>
    public DeviceSpan Input;

    /// <summary>Output.</summary>
    public DeviceSpan Output;

    /// <summary>Elements.</summary>
    public long Elements;

    /// <summary>GGUF tensor type id.</summary>
    public int GgufType;

    /// <summary>Activation element type.</summary>
    public DType DType;
}
