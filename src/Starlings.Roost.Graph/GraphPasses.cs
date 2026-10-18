using System.Collections.Generic;

namespace Starlings.Roost.Graph;

/// <summary>What a pass may assume about the target. Values come from the profile and the mesh (C9).</summary>
public sealed class GraphTarget
{
    /// <summary>Creates a target.</summary>
    /// <param name="sm">Compute capability as major * 10 + minor; 0 for the CPU reference.</param>
    /// <param name="tensorParallel">Tensor-parallel size.</param>
    /// <param name="expertParallel">Expert-parallel size.</param>
    /// <param name="pipelineParallel">Pipeline-parallel size.</param>
    /// <param name="availableVariants">Conformance-passed variant ids by op name, from the loaded registration tables.</param>
    public GraphTarget(int sm, int tensorParallel, int expertParallel, int pipelineParallel, IReadOnlyDictionary<string, IReadOnlyList<string>> availableVariants)
    {
        Sm = sm;
        TensorParallel = tensorParallel;
        ExpertParallel = expertParallel;
        PipelineParallel = pipelineParallel;
        AvailableVariants = availableVariants;
    }

    /// <summary>Compute capability, or 0.</summary>
    public int Sm { get; }

    /// <summary>Tensor-parallel size.</summary>
    public int TensorParallel { get; }

    /// <summary>Expert-parallel size.</summary>
    public int ExpertParallel { get; }

    /// <summary>Pipeline-parallel size.</summary>
    public int PipelineParallel { get; }

    /// <summary>Conformance-passed variants by op name.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> AvailableVariants { get; }
}

/// <summary>The kinds of pass (build plan section 3.1).</summary>
public enum PassKind
{
    /// <summary>Replaces op sequences by optional fused ops.</summary>
    Fusion = 0,

    /// <summary>Adds sharding annotations and collectives.</summary>
    Sharding,

    /// <summary>Chooses a kernel variant per node.</summary>
    VariantSelection,

    /// <summary>Lowers a whole step to a persistent kernel.</summary>
    PersistentLowering,
}

/// <summary>A graph-to-graph pass. A pass never mutates its input.</summary>
public interface IGraphPass
{
    /// <summary>Stable name, recorded in evidence.</summary>
    string Name { get; }

    /// <summary>The kind of pass.</summary>
    PassKind Kind { get; }

    /// <summary>Runs the pass.</summary>
    /// <param name="graph">The input graph.</param>
    /// <param name="target">The target.</param>
    /// <returns>A new graph whose every node's <see cref="OpNode.Origin"/> names reference-graph nodes.</returns>
    OpGraph Run(OpGraph graph, GraphTarget target);
}

/// <summary>Runs a graph on a backend and returns every node's outputs, for the checker.</summary>
/// <remarks>The CPU implementation runs the graph on <c>CpuOps</c> (stream D) in double precision.</remarks>
public interface IGraphEvaluator
{
    /// <summary>Evaluates every node.</summary>
    /// <param name="graph">The graph.</param>
    /// <param name="inputs">Input and weight values by value id, as float64.</param>
    /// <returns>Each written value id's contents as float64.</returns>
    IReadOnlyDictionary<int, double[]> Evaluate(OpGraph graph, IReadOnlyDictionary<int, double[]> inputs);
}

/// <summary>One node whose outputs disagree with its reference nodes.</summary>
public sealed class NodeMismatch
{
    /// <summary>Creates a mismatch record.</summary>
    /// <param name="candidateNode">Node id in the checked graph.</param>
    /// <param name="referenceNodes">Its origin nodes in the reference graph.</param>
    /// <param name="maxAbsoluteError">Largest absolute difference.</param>
    /// <param name="maxRelativeError">Largest relative difference.</param>
    public NodeMismatch(int candidateNode, IReadOnlyList<int> referenceNodes, double maxAbsoluteError, double maxRelativeError)
    {
        CandidateNode = candidateNode;
        ReferenceNodes = referenceNodes;
        MaxAbsoluteError = maxAbsoluteError;
        MaxRelativeError = maxRelativeError;
    }

    /// <summary>Node id in the checked graph.</summary>
    public int CandidateNode { get; }

    /// <summary>Origin nodes in the reference graph.</summary>
    public IReadOnlyList<int> ReferenceNodes { get; }

    /// <summary>Largest absolute difference.</summary>
    public double MaxAbsoluteError { get; }

    /// <summary>Largest relative difference.</summary>
    public double MaxRelativeError { get; }
}

/// <summary>Outcome of a check.</summary>
public sealed class GraphCheckResult
{
    /// <summary>Creates a result.</summary>
    /// <param name="nodesChecked">Nodes compared.</param>
    /// <param name="mismatches">Nodes outside tolerance.</param>
    public GraphCheckResult(int nodesChecked, IReadOnlyList<NodeMismatch> mismatches)
    {
        NodesChecked = nodesChecked;
        Mismatches = mismatches;
    }

    /// <summary>Nodes compared.</summary>
    public int NodesChecked { get; }

    /// <summary>Nodes outside tolerance.</summary>
    public IReadOnlyList<NodeMismatch> Mismatches { get; }

    /// <summary>True when every node is within tolerance.</summary>
    public bool Passed => Mismatches.Count == 0;
}

/// <summary>Per-op-class tolerances for the checker; the values live in maintainer-owned files (evidence/protected).</summary>
public interface IToleranceSource
{
    /// <summary>Tolerance for a node.</summary>
    /// <param name="op">The C3 op name of the checked node.</param>
    /// <param name="dtype">Its output element type.</param>
    /// <returns>Absolute and relative tolerance.</returns>
    (double Absolute, double Relative) Lookup(string op, GraphDType dtype);
}

/// <summary>
/// The C7 rule: every pass output is checked op by op against the unfused reference graph on <c>CpuOps</c>.
/// </summary>
public interface IGraphChecker
{
    /// <summary>Checks <paramref name="candidate"/> against <paramref name="reference"/>.</summary>
    /// <param name="reference">The unfused reference graph (<see cref="OpGraph.IsReference"/>).</param>
    /// <param name="candidate">A pass output.</param>
    /// <param name="evaluator">The evaluator, normally <c>CpuOps</c>.</param>
    /// <param name="inputs">Inputs and weights by reference-graph value id.</param>
    /// <param name="tolerances">Tolerances.</param>
    /// <returns>The outcome.</returns>
    GraphCheckResult Check(
        OpGraph reference,
        OpGraph candidate,
        IGraphEvaluator evaluator,
        IReadOnlyDictionary<int, double[]> inputs,
        IToleranceSource tolerances);
}
