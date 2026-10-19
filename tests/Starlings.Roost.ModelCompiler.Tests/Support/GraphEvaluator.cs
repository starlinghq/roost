using System.Globalization;
using System.Reflection;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Graph;
using Starlings.Roost.Ops;

namespace Starlings.Roost.ModelCompiler.Tests;

/// <summary>
/// Evaluates an OpGraph by running each node on <typeparamref name="TOps"/>. A node's argument struct is built by
/// reflection from the graph alone: the <c>in</c> and <c>out</c> attributes name the field of each input and output
/// position and <c>arg.FIELD</c> holds the scalars, which is the same information the generated Forward uses. Test-time
/// code only (reflection is not AOT-safe). Every value gets its own buffer, so every intermediate can be read back.
/// </summary>
public sealed unsafe class GraphEvaluator<TOps> : IGraphEvaluator
    where TOps : struct, IRoostOps
{
    private readonly Dictionary<string, long> _symbols;

    public GraphEvaluator(IReadOnlyDictionary<string, long> symbols) => _symbols = new Dictionary<string, long>(symbols);

    public IReadOnlyDictionary<int, double[]> Evaluate(OpGraph graph, IReadOnlyDictionary<int, double[]> inputs)
    {
        using var native = new Native();
        var spans = new Dictionary<int, DeviceSpan>();
        foreach (var v in graph.Values)
        {
            long n = 1;
            foreach (var d in v.Shape)
            {
                n *= d.IsSymbolic ? Symbol(d.Symbol!) : d.Size;
            }

            var span = native.Alloc(n, ToAbi(v.DType));
            spans[v.Id] = span;
            if (inputs.TryGetValue(v.Id, out double[]? data))
            {
                Write(span, data);
            }
        }

        foreach (var node in graph.Nodes)
        {
            Run(node, spans);
        }

        var result = new Dictionary<int, double[]>();
        foreach (var v in graph.Values)
        {
            if (v.Kind is ValueKind.Intermediate or ValueKind.Output)
            {
                result[v.Id] = Read(spans[v.Id]);
            }
        }

        return result;
    }

    private long Symbol(string text) =>
        DimExpression.TryEvaluate(text, _symbols, out long v, out _) ? v : 1;

    private void Run(OpNode node, Dictionary<int, DeviceSpan> spans)
    {
        var method = typeof(TOps).GetMethod(node.Op, BindingFlags.Public | BindingFlags.Static)
            ?? throw new NotSupportedException(node.Op);
        var supports = typeof(TOps).GetProperty("Supports" + node.Op, BindingFlags.Public | BindingFlags.Static);
        if (supports is not null && !(bool)supports.GetValue(null)!)
        {
            throw new NotSupportedException(node.Op + " is a fused op the backend does not provide");
        }

        var argsType = method.GetParameters()[0].ParameterType.GetElementType()!;
        object args = Activator.CreateInstance(argsType)!;
        string[] inFields = Fields(node, "in");
        string[] outFields = Fields(node, "out");

        for (int o = 0; o < outFields.Length; o++)
        {
            var span = spans[node.Outputs[o]];
            int i = Array.IndexOf(inFields, outFields[o]);
            if (i >= 0)
            {
                var src = spans[node.Inputs[i]];
                Buffer.MemoryCopy((void*)src.Address, (void*)span.Address, (long)span.Length * Native.Size(span.Dtype), (long)src.Length * Native.Size(src.Dtype));
            }

            argsType.GetField(outFields[o])!.SetValue(args, span);
        }

        for (int i = 0; i < inFields.Length; i++)
        {
            if (Array.IndexOf(outFields, inFields[i]) < 0)
            {
                argsType.GetField(inFields[i])!.SetValue(args, spans[node.Inputs[i]]);
            }
        }

        foreach (var (key, text) in node.Attributes)
        {
            if (!key.StartsWith("arg.", StringComparison.Ordinal))
            {
                continue;
            }

            var field = argsType.GetField(key.Substring(4))!;
            object value = field.FieldType switch
            {
                var t when t == typeof(int) => (int)DimExpression.Evaluate(text, _symbols),
                var t when t == typeof(long) => DimExpression.Evaluate(text, _symbols),
                var t when t == typeof(float) => float.Parse(text, CultureInfo.InvariantCulture),
                var t when t == typeof(bool) => bool.Parse(text),
                var t => Enum.Parse(t, text),
            };
            field.SetValue(args, value);
        }

        try
        {
            method.Invoke(null, [args]);
        }
        catch (TargetInvocationException e) when (e.InnerException is not null)
        {
            throw e.InnerException;
        }
    }

    private static string[] Fields(OpNode node, string key) =>
        node.Attributes[key].Length == 0 ? [] : node.Attributes[key].Split(',');

    private static DType ToAbi(GraphDType t) => (DType)(uint)t;

    private static void Write(DeviceSpan span, double[] data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            switch (span.Dtype)
            {
                case DType.F32:
                    ((float*)span.Address)[i] = (float)data[i];
                    break;
                case DType.I32:
                    ((int*)span.Address)[i] = (int)data[i];
                    break;
                case DType.I64:
                    ((long*)span.Address)[i] = (long)data[i];
                    break;
                case DType.F64:
                    ((double*)span.Address)[i] = data[i];
                    break;
                default:
                    throw new NotSupportedException(span.Dtype.ToString());
            }
        }
    }

    private static double[] Read(DeviceSpan span)
    {
        var r = new double[span.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = span.Dtype switch
            {
                DType.F32 => ((float*)span.Address)[i],
                DType.I32 => ((int*)span.Address)[i],
                DType.I64 => ((long*)span.Address)[i],
                DType.F64 => ((double*)span.Address)[i],
                _ => throw new NotSupportedException(span.Dtype.ToString()),
            };
        }

        return r;
    }
}
