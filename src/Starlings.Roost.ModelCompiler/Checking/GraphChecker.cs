using System;
using System.Collections.Generic;
using Starlings.Roost.Graph;

namespace Starlings.Roost.ModelCompiler.Checking;

/// <summary>
/// The C7 checker: runs the unfused reference graph and a pass's output graph through the same evaluator (on
/// <c>CpuOps</c> in the real use) and compares, node by node, every value a candidate node produces with the
/// reference value of the same id. A node that replaces several reference nodes (see <see cref="OpNode.Origin"/>) is
/// compared against the values those nodes produced. Values a candidate no longer materialises are not compared; a
/// model output the candidate fails to produce is a mismatch.
/// </summary>
public sealed class GraphChecker : IGraphChecker
{
    /// <summary>Runs <paramref name="pass"/> over the unfused <paramref name="reference"/> graph and checks its output (the C7 rule for every pass).</summary>
    /// <param name="pass">The pass under test.</param>
    /// <param name="reference">The unfused graph (<see cref="OpGraph.IsReference"/> is true).</param>
    /// <param name="target">What the pass may assume about the target.</param>
    /// <param name="evaluator">Runs a graph; on <c>CpuOps</c> for the real check.</param>
    /// <param name="inputs">Values of the weight, input and state values, by value id.</param>
    /// <param name="tolerances">Per-op and per-dtype tolerances.</param>
    /// <returns>The comparison of the pass output with the reference.</returns>
    public GraphCheckResult CheckPass(
        IGraphPass pass,
        OpGraph reference,
        GraphTarget target,
        IGraphEvaluator evaluator,
        IReadOnlyDictionary<int, double[]> inputs,
        IToleranceSource tolerances)
    {
        if (pass is null)
        {
            throw new ArgumentNullException(nameof(pass));
        }

        if (reference is null)
        {
            throw new ArgumentNullException(nameof(reference));
        }

        if (!reference.IsReference)
        {
            throw new ArgumentException("The reference must be the unfused graph (OpGraph.IsReference).", nameof(reference));
        }

        return Check(reference, pass.Run(reference, target), evaluator, inputs, tolerances);
    }

    /// <inheritdoc/>
    public GraphCheckResult Check(
        OpGraph reference,
        OpGraph candidate,
        IGraphEvaluator evaluator,
        IReadOnlyDictionary<int, double[]> inputs,
        IToleranceSource tolerances)
    {
        if (reference is null)
        {
            throw new ArgumentNullException(nameof(reference));
        }

        if (candidate is null)
        {
            throw new ArgumentNullException(nameof(candidate));
        }

        if (evaluator is null)
        {
            throw new ArgumentNullException(nameof(evaluator));
        }

        if (tolerances is null)
        {
            throw new ArgumentNullException(nameof(tolerances));
        }

        var expected = evaluator.Evaluate(reference, inputs);
        var actual = evaluator.Evaluate(candidate, inputs);
        var mismatches = new List<NodeMismatch>();
        var produced = new HashSet<int>();
        int checkedNodes = 0;

        foreach (var node in candidate.Nodes)
        {
            checkedNodes++;
            double maxAbs = 0, maxRel = 0;
            bool failed = false;
            foreach (int origin in node.Origin)
            {
                if (origin < 0 || origin >= reference.Nodes.Count)
                {
                    // A node that claims to replace a reference node that does not exist cannot be checked op by op.
                    failed = true;
                    maxAbs = double.PositiveInfinity;
                    maxRel = double.PositiveInfinity;
                }
            }

            foreach (int id in node.Outputs)
            {
                produced.Add(id);
                if (!expected.TryGetValue(id, out double[]? want) || !actual.TryGetValue(id, out double[]? got))
                {
                    failed = true;
                    maxAbs = double.PositiveInfinity;
                    maxRel = double.PositiveInfinity;
                    continue;
                }

                var (abs, rel) = tolerances.Lookup(node.Op, candidate.Values[id].DType);
                if (!Compare(want, got, abs, rel, ref maxAbs, ref maxRel))
                {
                    failed = true;
                }
            }

            if (failed)
            {
                mismatches.Add(new NodeMismatch(node.Id, node.Origin, maxAbs, maxRel));
            }
        }

        foreach (var v in reference.Values)
        {
            if (v.Kind == ValueKind.Output && !produced.Contains(v.Id))
            {
                mismatches.Add(new NodeMismatch(-1, Array.Empty<int>(), double.PositiveInfinity, double.PositiveInfinity));
            }
        }

        return new GraphCheckResult(checkedNodes, mismatches);
    }

    private static bool Compare(double[] want, double[] got, double absTol, double relTol, ref double maxAbs, ref double maxRel)
    {
        if (want.Length != got.Length)
        {
            maxAbs = double.PositiveInfinity;
            maxRel = double.PositiveInfinity;
            return false;
        }

        bool ok = true;
        for (int i = 0; i < want.Length; i++)
        {
            double w = want[i], g = got[i];
            if (double.IsNaN(w) && double.IsNaN(g))
            {
                continue;
            }

            if (double.IsNaN(w) || double.IsNaN(g))
            {
                maxAbs = double.PositiveInfinity;
                maxRel = double.PositiveInfinity;
                ok = false;
                continue;
            }

            if (w.Equals(g))
            {
                continue;
            }

            double diff = Math.Abs(w - g);
            double rel = diff / Math.Max(Math.Abs(w), double.Epsilon);
            maxAbs = Math.Max(maxAbs, diff);
            maxRel = Math.Max(maxRel, rel);
            if (!(diff <= absTol + (relTol * Math.Abs(w))))
            {
                ok = false;
            }
        }

        return ok;
    }
}
