using System;

namespace Spike.Ops;

/// <summary>Reference CPU backend: float math over raw buffers, no allocation. Stands in for stream D's CpuOps.</summary>
public unsafe struct CpuOps : IRoostOps
{
    private const float Eps = 1e-6f;

    public static void Embedding(in EmbeddingArgs a)
    {
        int* ids = (int*)a.Input.Ptr;
        float* table = (float*)a.Weight.Ptr;
        float* o = (float*)a.Output.Ptr;
        for (int r = 0; r < a.Rows; r++)
        {
            Buffer.MemoryCopy(table + ((long)ids[r] * a.OutDim), o + ((long)r * a.OutDim), a.OutDim * 4L, a.OutDim * 4L);
        }
    }

    public static void RmsNorm(in RmsNormArgs a) => RmsNormCore((float*)a.Input.Ptr, (float*)a.Weight.Ptr, (float*)a.Output.Ptr, a.Rows, a.InDim);

    internal static void RmsNormCore(float* x, float* w, float* o, int rows, int dim)
    {
        for (int r = 0; r < rows; r++)
        {
            float* xr = x + ((long)r * dim);
            float ss = 0;
            for (int i = 0; i < dim; i++)
            {
                ss += xr[i] * xr[i];
            }

            float inv = 1f / MathF.Sqrt((ss / dim) + Eps);
            for (int i = 0; i < dim; i++)
            {
                o[((long)r * dim) + i] = xr[i] * inv * w[i];
            }
        }
    }

    public static void Gemm(in GemmArgs a) => GemmCore((float*)a.Input.Ptr, (float*)a.Weight.Ptr, (float*)a.Output.Ptr, a.Rows, a.InDim, a.OutDim);

    public static void LmHead(in LmHeadArgs a) => GemmCore((float*)a.Input.Ptr, (float*)a.Weight.Ptr, (float*)a.Output.Ptr, a.Rows, a.InDim, a.OutDim);

    private static void GemmCore(float* x, float* w, float* o, int rows, int k, int n)
    {
        for (int r = 0; r < rows; r++)
        {
            for (int j = 0; j < n; j++)
            {
                float acc = 0;
                float* wj = w + ((long)j * k);
                float* xr = x + ((long)r * k);
                for (int i = 0; i < k; i++)
                {
                    acc += xr[i] * wj[i];
                }

                o[((long)r * n) + j] = acc;
            }
        }
    }

    public static void SiluAndMul(in SiluAndMulArgs a)
    {
        float* x = (float*)a.Input.Ptr;
        float* o = (float*)a.Output.Ptr;
        int d = a.OutDim;
        for (int r = 0; r < a.Rows; r++)
        {
            for (int i = 0; i < d; i++)
            {
                float g = x[((long)r * 2 * d) + i];
                float u = x[((long)r * 2 * d) + d + i];
                o[((long)r * d) + i] = g / (1f + MathF.Exp(-g)) * u;
            }
        }
    }

    public static void Residual(in ResidualArgs a)
    {
        float* x = (float*)a.Input.Ptr;
        float* y = (float*)a.Input2.Ptr;
        float* o = (float*)a.Output.Ptr;
        long n = (long)a.Rows * a.InDim;
        for (long i = 0; i < n; i++)
        {
            o[i] = x[i] + y[i];
        }
    }
}

/// <summary>A backend that provides the optional fused AddRmsNorm, to exercise the fused lowering.</summary>
public unsafe struct FusedCpuOps : IRoostOps
{
    public static bool SupportsAddRmsNorm => true;

    public static void AddRmsNorm(in AddRmsNormArgs a)
    {
        var add = new ResidualArgs { Input = a.Input, Input2 = a.Input2, Output = a.Output2, Rows = a.Rows, InDim = a.InDim };
        CpuOps.Residual(in add);
        CpuOps.RmsNormCore((float*)a.Output2.Ptr, (float*)a.Weight.Ptr, (float*)a.Output.Ptr, a.Rows, a.InDim);
    }

    public static void Embedding(in EmbeddingArgs a) => CpuOps.Embedding(in a);
    public static void RmsNorm(in RmsNormArgs a) => CpuOps.RmsNorm(in a);
    public static void Gemm(in GemmArgs a) => CpuOps.Gemm(in a);
    public static void SiluAndMul(in SiluAndMulArgs a) => CpuOps.SiluAndMul(in a);
    public static void Residual(in ResidualArgs a) => CpuOps.Residual(in a);
    public static void LmHead(in LmHeadArgs a) => CpuOps.LmHead(in a);
}
