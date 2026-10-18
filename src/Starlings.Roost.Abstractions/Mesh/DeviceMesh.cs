using System.Collections.Immutable;
using System.Runtime.CompilerServices;

namespace Starlings.Roost.Mesh;

/// <summary>One rank of the mesh: one worker process driving one GPU (contract C9).</summary>
/// <param name="Rank">Global rank.</param>
/// <param name="Node">Host index.</param>
/// <param name="DeviceOrdinal">CUDA device ordinal on that host.</param>
/// <param name="TensorRank">Rank within the tensor-parallel group.</param>
/// <param name="PipelineRank">Pipeline stage.</param>
/// <param name="DataRank">Data-parallel replica.</param>
/// <param name="ExpertRank">Rank within the expert-parallel group.</param>
public sealed record MeshRank(int Rank, int Node, int DeviceOrdinal, int TensorRank, int PipelineRank, int DataRank, int ExpertRank);

/// <summary>The parallel sizes of a mesh, as recorded in a profile manifest.</summary>
/// <param name="TensorParallel">TP size.</param>
/// <param name="PipelineParallel">PP size.</param>
/// <param name="DataParallel">DP size.</param>
/// <param name="ExpertParallel">EP size; experts are split over EP ranks drawn from the TP x DP ranks.</param>
public sealed record MeshDescriptor(int TensorParallel, int PipelineParallel, int DataParallel, int ExpertParallel)
{
    /// <summary>One GPU.</summary>
    public static MeshDescriptor SingleRank { get; } = new(1, 1, 1, 1);

    /// <summary>Ranks: TP x PP x DP. EP reuses those ranks.</summary>
    public int WorldSize => TensorParallel * PipelineParallel * DataParallel;
}

/// <summary>The NCCL unique id (128 bytes) that rank 0 creates and the launcher distributes for bootstrap.</summary>
[InlineArray(Size)]
public struct NcclUniqueId
{
    /// <summary>Bytes in an NCCL unique id (<c>NCCL_UNIQUE_ID_BYTES</c>).</summary>
    public const int Size = 128;

    private byte _element0;
}

/// <summary>A device mesh: ranks and parallel sizes. One worker process per GPU; each rank gets its own step plans.</summary>
public sealed record DeviceMesh
{
    /// <summary>Mesh id, carried in per-rank plan headers.</summary>
    public required uint MeshId { get; init; }

    /// <summary>Parallel sizes.</summary>
    public required MeshDescriptor Descriptor { get; init; }

    /// <summary>Ranks, indexed by global rank.</summary>
    public required ImmutableArray<MeshRank> Ranks { get; init; }

    /// <summary>The single-rank mesh used through M3.</summary>
    /// <param name="deviceOrdinal">CUDA device ordinal.</param>
    /// <returns>A mesh with one rank.</returns>
    public static DeviceMesh SingleRank(int deviceOrdinal = 0) => new()
    {
        MeshId = 0,
        Descriptor = MeshDescriptor.SingleRank,
        Ranks = [new MeshRank(0, 0, deviceOrdinal, 0, 0, 0, 0)],
    };
}
