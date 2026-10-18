// Spike S10 harness: loads libroost_grammar through P/Invoke from a Native AOT binary,
// replays a valid JSON instance token by token, and times each mask computation.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

internal static unsafe partial class Native
{
    private const string Lib = "roost_grammar";

    [DllImport(Lib)] internal static extern uint roost_grammar_abi_version();
    [DllImport(Lib)] internal static extern nuint roost_grammar_last_error(byte* buf, nuint cap);
    [DllImport(Lib)] internal static extern void* roost_grammar_tokenizer_new(byte* json, nuint len, uint nVocab);
    [DllImport(Lib)] internal static extern void roost_grammar_tokenizer_free(void* t);
    [DllImport(Lib)] internal static extern nuint roost_grammar_mask_words(void* t);
    [DllImport(Lib)] internal static extern nint roost_grammar_tokenize(void* t, byte* text, nuint len, uint* outp, nuint cap);
    [DllImport(Lib)] internal static extern uint roost_grammar_eos_token(void* t);
    [DllImport(Lib)] internal static extern void* roost_grammar_matcher_new(void* t, byte* kind, byte* data);
    [DllImport(Lib)] internal static extern void roost_grammar_matcher_free(void* m);
    [DllImport(Lib)] internal static extern int roost_grammar_matcher_compute_mask(void* m, uint* dest, nuint words);
    [DllImport(Lib)] internal static extern int roost_grammar_matcher_consume(void* m, uint token);
    [DllImport(Lib)] internal static extern int roost_grammar_matcher_is_accepting(void* m);
    [DllImport(Lib)] internal static extern int roost_grammar_matcher_is_stopped(void* m);
}

internal static unsafe class Program
{
    private static string LastError()
    {
        byte* buf = stackalloc byte[512];
        Native.roost_grammar_last_error(buf, 512);
        return Marshal.PtrToStringUTF8((nint)buf) ?? "";
    }

    private static byte[] Z(string s) => Encoding.UTF8.GetBytes(s + "\0");

    private static double Pct(double[] sorted, double p)
    {
        int i = (int)Math.Ceiling(p / 100.0 * sorted.Length) - 1;
        return sorted[Math.Clamp(i, 0, sorted.Length - 1)];
    }

    private static int Main(string[] args)
    {
        // args: tokenizer.json schemaDir reps nVocab
        string tokPath = args[0], dir = args[1];
        int reps = int.Parse(args[2]);
        uint nVocab = uint.Parse(args[3]);
        Console.WriteLine($"abi_version={Native.roost_grammar_abi_version()} reps={reps} n_vocab={nVocab}");

        byte[] tokJson = File.ReadAllBytes(tokPath);
        long t0 = Stopwatch.GetTimestamp();
        void* tok;
        fixed (byte* p = tokJson) tok = Native.roost_grammar_tokenizer_new(p, (nuint)tokJson.Length, nVocab);
        if (tok == null) { Console.WriteLine("tokenizer_new failed: " + LastError()); return 1; }
        Console.WriteLine($"tokenizer_new_ms={Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F1}");
        int words = (int)Native.roost_grammar_mask_words(tok);
        uint eos = Native.roost_grammar_eos_token(tok);
        Console.WriteLine($"mask_words={words} mask_bytes={words * 4} eos={eos}");
        uint* mask = (uint*)NativeMemory.AlignedAlloc((nuint)(words * 4), 64);

        var results = new List<string>();
        foreach (string schemaPath in Directory.GetFiles(dir, "*.json").Where(f => !f.EndsWith(".instance.json")).OrderBy(f => f))
        {
            string name = Path.GetFileNameWithoutExtension(schemaPath);
            string schema = File.ReadAllText(schemaPath).Trim();
            byte[] inst = Encoding.UTF8.GetBytes(File.ReadAllText(Path.ChangeExtension(schemaPath, ".instance.json")).Trim());
            uint[] toks = new uint[inst.Length + 1];
            nint n;
            fixed (byte* ip = inst) fixed (uint* tp = toks) n = Native.roost_grammar_tokenize(tok, ip, (nuint)inst.Length, tp, (nuint)toks.Length);
            if (n < 0) { Console.WriteLine("tokenize failed: " + LastError()); return 1; }
            toks[n] = eos;
            int steps = (int)n + 1;

            var maskNs = new List<double>(reps * steps);
            var coldFirstNs = new List<double>();
            var newMs = new List<double>();
            var consumeNs = new List<double>(reps * steps);
            double firstRepMax = 0;
            int violations = 0;
            byte[] kind = Z("json_schema"), data = Z(schema);
            for (int r = 0; r < reps; r++)
            {
                long a = Stopwatch.GetTimestamp();
                void* m;
                fixed (byte* kp = kind) fixed (byte* dp = data) m = Native.roost_grammar_matcher_new(tok, kp, dp);
                if (m == null) { Console.WriteLine($"{name}: matcher_new failed: " + LastError()); return 1; }
                newMs.Add(Stopwatch.GetElapsedTime(a).TotalMilliseconds);
                for (int i = 0; i < steps; i++)
                {
                    long b = Stopwatch.GetTimestamp();
                    int rc = Native.roost_grammar_matcher_compute_mask(m, mask, (nuint)words);
                    long c = Stopwatch.GetTimestamp();
                    if (rc != 0) { Console.WriteLine($"{name}: mask failed at {i}: " + LastError()); return 1; }
                    double ns = (c - b) * 1e9 / Stopwatch.Frequency;
                    maskNs.Add(ns);
                    if (r == 0) firstRepMax = Math.Max(firstRepMax, ns);
                    uint t = toks[i];
                    if ((mask[t >> 5] & (1u << (int)(t & 31))) == 0) { violations++; Console.WriteLine($"{name}: token {t} at step {i} not allowed by mask"); break; }
                    if (t == eos) continue; // the matcher has already stopped: a closed JSON value only allows EOS
                    long d = Stopwatch.GetTimestamp();
                    rc = Native.roost_grammar_matcher_consume(m, t);
                    long e = Stopwatch.GetTimestamp();
                    if (rc != 0) { Console.WriteLine($"{name}: consume failed at {i}: " + LastError()); return 1; }
                    consumeNs.Add((e - d) * 1e9 / Stopwatch.Frequency);
                }
                if (r == 0)
                {
                    int acc = Native.roost_grammar_matcher_is_accepting(m), stop = Native.roost_grammar_matcher_is_stopped(m);
                    Console.WriteLine($"{name}: accepting={acc} stopped={stop}");
                }
                Native.roost_grammar_matcher_free(m);
            }
            double[] s = maskNs.ToArray(); Array.Sort(s);
            double[] cs = consumeNs.ToArray(); Array.Sort(cs);
            double[] nm = newMs.ToArray(); Array.Sort(nm);
            string line = $"{name}: instance_bytes={inst.Length} tokens={n} steps={steps} samples={s.Length} violations={violations} " +
                $"mask_us p50={Pct(s, 50) / 1e3:F1} p90={Pct(s, 90) / 1e3:F1} p99={Pct(s, 99) / 1e3:F1} p99.9={Pct(s, 99.9) / 1e3:F1} max={s[^1] / 1e3:F1} mean={s.Average() / 1e3:F1} " +
                $"cold_rep0_max_us={firstRepMax / 1e3:F1} consume_us p50={Pct(cs, 50) / 1e3:F1} p99={Pct(cs, 99) / 1e3:F1} " +
                $"matcher_new_ms first={newMs[0]:F2} p50={Pct(nm, 50):F2} max={nm[^1]:F2}";
            Console.WriteLine(line);
        }
        NativeMemory.AlignedFree(mask);
        Native.roost_grammar_tokenizer_free(tok);
        return 0;
    }
}
