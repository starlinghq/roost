using Starlings.Roost.Abi;

namespace Roost.KernelOps.Tests.Cuda;

/// <summary>What a case may use: the loaded library, a stream, a seeded generator and the result log.</summary>
internal sealed class Ctx
{
    private readonly List<CompareResult> _results = [];
    private readonly TextWriter _log;

    public Ctx(Library lib, nint stream, TextWriter log, int smCurrent)
    {
        Lib = lib;
        Stream = stream;
        _log = log;
        Sm = smCurrent;
    }

    public Library Lib { get; }

    public nint Stream { get; }

    public int Sm { get; }

    public Rng Rng { get; set; } = new(1);

    public IReadOnlyList<CompareResult> Results => _results;

    public void Report(CompareResult r)
    {
        _results.Add(r);
        Line(r.ToString());
    }

    public void Check(string name, bool ok, string detail = "")
    {
        var r = new CompareResult(name, "check", 1, 0, 0, ok ? 0 : 1, 0, 0, ok);
        _results.Add(r);
        Line($"{(ok ? "PASS" : "FAIL")} {name,-58} {detail}");
    }

    public void Line(string s)
    {
        _log.WriteLine(s);
        _log.Flush();
        Console.WriteLine(s);
    }

    /// <summary>Copies a device buffer's contents to a fresh host-side buffer of the same type.</summary>
    public static Buf Fetch(Buf deviceSide)
    {
        var host = new Buf(deviceSide.Dtype, deviceSide.Count);
        Cuda.ToHost(host.Host, deviceSide.Dev, deviceSide.Bytes);
        return host;
    }

    /// <summary>Rows of <paramref name="width"/> elements out of a buffer with row stride <paramref name="stride"/>.</summary>
    public static Buf Compact(Buf strided, long rows, long width, long stride)
    {
        var c = new Buf(strided.Dtype, rows * width);
        int es = Elem.Size(strided.Dtype);
        for (long r = 0; r < rows; r++)
        {
            unsafe
            {
                Buffer.MemoryCopy((void*)(strided.Host + (nint)(r * stride * es)), (void*)(c.Host + (nint)(r * width * es)), width * es, width * es);
            }
        }

        return c;
    }

    public static int StructSize<T>() where T : unmanaged
    {
        unsafe
        {
            return sizeof(T);
        }
    }
}
