using System.Runtime.InteropServices;
using System.Text;
using Starlings.Roost;
using Starlings.Roost.Abi;
using Starlings.Roost.Cuda;
using Starlings.Roost.Kernels;
using Starlings.Roost.Requests;

// R-023 GPU evidence: the R-016 stub library (built by kernels/CMakeLists.txt, never committed) loaded through the
// loader on a real device, then run. Everything here returns statuses or throws managed exceptions; nothing raises a
// signal or launches a faulting kernel.
//   usage: kernels-loader-check <kernel-build-dir> <manifest.toml> [--allow-untested]
// Run it through tools/gpu-lease so CUDA_VISIBLE_DEVICES selects the leased device (ordinal 0 here).
if (args.Length < 2)
{
    Console.Error.WriteLine("usage: kernels-loader-check <kernel-build-dir> <manifest.toml> [--allow-untested]");
    return 2;
}

return Check.Run(args[0], args[1], args.Contains("--allow-untested"));

internal static unsafe class Check
{
    private static int _failures;

    public static int Run(string kernelDir, string manifestPath, bool allowUntested)
    {
        var manifest = KernelManifest.Load(manifestPath);
        Console.WriteLine($"manifest sha256 {manifest.Sha256}, abi {manifest.AbiMajor}.{manifest.AbiMinor}, cuda {manifest.Cuda}, min driver {manifest.MinDriver}");

        // The per-OS resolver takes the CUDA major from the manifest, not from a constant.
        CudaLibraryResolver.Install(manifest.CudaMajor);

        int runtime, driver;
        CudaCheck.Runtime(CudaRuntime.cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        CudaCheck.Runtime(CudaRuntime.cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        int count;
        CudaCheck.Runtime(CudaRuntime.cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        var sm = CudaCheck.ComputeCapability(0);
        Console.WriteLine($"cudart {runtime}, driver {driver}, {count} visible device(s), device 0 is SM {sm}");
        Expect(runtime / 1000 == manifest.CudaMajor, $"cudart major {runtime / 1000} equals the manifest's {manifest.CudaMajor}");

        NvmlAndCublasLtSmoke();

        // 1. The typed refusals, on the real manifest and library. None of these loads the library.
        var file = KernelLibraryNames.Find(sm, [kernelDir], KernelLibraryNames.Current());
        Console.WriteLine($"library for SM {sm}: {file ?? "(none)"}");
        if (file is null)
        {
            Console.WriteLine($"no library for SM {sm} in {kernelDir}");
            return 3;
        }

        var tampered = Encoding.UTF8.GetString(File.ReadAllBytes(manifestPath));
        var digest = manifest.Variants.SelectMany(v => v.Binaries).First(b => b.Arch == sm).Sha256;
        var wrong = KernelManifest.Parse(Encoding.UTF8.GetBytes(tampered.Replace(digest, new string('0', 64), StringComparison.Ordinal)));
        Refused<KernelLoadException>("digest mismatch", () => KernelLibrary.Load(file, new KernelLoadOptions(wrong, sm, true)),
            e => e.Error.Code == RoostErrorCode.AssetMismatch);
        Refused<KernelLoadException>("manifest pin mismatch", () => KernelManifest.Load(manifestPath, new string('1', 64)),
            e => e.Error.Code == RoostErrorCode.AssetMismatch);
        Refused<ProfileUnsupportedException>("SM without a library (SM 75)", () => KernelLibrary.Load(file, new KernelLoadOptions(manifest, 75, true)),
            e => e.Sm == 75 && e.Op == OpId.Residual);
        Refused<ProfileUnsupportedException>("untested SM 89 without opt-in", () => KernelLibrary.Load(file, new KernelLoadOptions(manifest, 89)),
            e => e.Sm == 89);

        // 2. The load itself.
        using var library = KernelLibrary.Load(file, new KernelLoadOptions(manifest, sm, allowUntested));
        Console.WriteLine($"loaded {library.Name}: abi {library.AbiVersion >> 16}.{library.AbiVersion & 0xFFFF}, SM status {library.SmStatus}, {library.Kernels.Count} entries");
        foreach (var k in library.Kernels)
        {
            Console.WriteLine($"  op {k.Op} variant {k.VariantId} flags {k.Flags} sm_min {k.Constraints.SmMin}");
        }

        var registry = new KernelRegistry();
        registry.Add(library);
        var stub = registry.Require(new KernelRequirement { Op = OpId.Residual, Sm = sm, Activation = DType.BF16, Layout = TensorLayout.RowMajor });
        Expect(stub.VariantId == "residual-stub", "Residual bf16 resolves to residual-stub");
        Refused<ProfileUnsupportedException>("missing specialisation (Residual f8e4m3)",
            () => registry.Require(new KernelRequirement { Op = OpId.Residual, Sm = sm, Activation = DType.F8E4M3 }),
            e => e.Op == OpId.Residual && e.Sm == sm && e.Message.Contains("F8E4M3", StringComparison.Ordinal));
        Refused<ProfileUnsupportedException>("missing op (PagedAttention)",
            () => registry.Require(new KernelRequirement { Op = OpId.PagedAttention, Sm = sm, HeadDim = 256 }),
            e => e.Op == OpId.PagedAttention && e.Sm == sm);

        // 3. Run the stub: bf16, an element count that is not a multiple of the block size, output aliasing a.
        CudaCheck.Runtime(CudaRuntime.cudaSetDevice(0), "cudaSetDevice");
        using var context = library.CreateContext(0);
        nint stream;
        CudaCheck.Runtime(CudaRuntime.cudaStreamCreateWithFlags(&stream, CudaRuntime.cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        const int n = 100003;
        var ha = new ushort[n];
        var hb = new ushort[n];
        var expect = new float[n];
        for (var i = 0; i < n; i++)
        {
            ha[i] = ToBf16(((i % 977) - 488) * 0.01f);
            hb[i] = ToBf16(((i % 613) - 306) * 0.02f);
            expect[i] = FromBf16(ToBf16(FromBf16(ha[i]) + FromBf16(hb[i])));
        }

        void* da, db;
        CudaCheck.Runtime(CudaRuntime.cudaMalloc(&da, n * 2), "cudaMalloc a");
        CudaCheck.Runtime(CudaRuntime.cudaMalloc(&db, n * 2), "cudaMalloc b");
        fixed (ushort* pa = ha, pb = hb)
        {
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy(da, pa, n * 2, CudaRuntime.cudaMemcpyHostToDevice), "memcpy a");
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy(db, pb, n * 2, CudaRuntime.cudaMemcpyHostToDevice), "memcpy b");
        }

        var p = new ResidualParams
        {
            StructSize = (uint)sizeof(ResidualParams),
            AbiMinor = (uint)KernelAbi.Minor,
            A = new DeviceSpan { Address = (nint)da, Length = n, Dtype = DType.BF16 },
            B = new DeviceSpan { Address = (nint)db, Length = n, Dtype = DType.BF16 },
            Output = new DeviceSpan { Address = (nint)da, Length = n, Dtype = DType.BF16 },
            Elements = n,
        };

        // Rejections are statuses with a message, never a launch.
        var bad = p;
        bad.B.Dtype = DType.F32;
        Expect(stub.RunKernel(context, &bad, null, stream) == KernelStatus.InvalidArgument, "dtype mismatch returns InvalidArgument");
        bad = p;
        bad.StructSize = 16;
        Expect(stub.RunKernel(context, &bad, null, stream) == KernelStatus.AbiMismatch, "short params struct returns AbiMismatch");

        var status = stub.RunKernel(context, &p, null, stream);
        Expect(status == KernelStatus.Ok, $"run returned {status}");
        CudaCheck.Runtime(CudaRuntime.cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        var hout = new ushort[n];
        fixed (ushort* po = hout)
        {
            CudaCheck.Runtime(CudaRuntime.cudaMemcpy(po, da, n * 2, CudaRuntime.cudaMemcpyDeviceToHost), "memcpy out");
        }

        var mismatches = 0;
        for (var i = 0; i < n; i++)
        {
            if (FromBf16(hout[i]) != expect[i])
            {
                mismatches++;
            }
        }

        Console.WriteLine($"Residual bf16, {n} elements, output aliasing a: {mismatches} mismatches");
        Expect(mismatches == 0, "output equals the host reference");

        CudaCheck.Runtime(CudaRuntime.cudaFree(da), "cudaFree a");
        CudaCheck.Runtime(CudaRuntime.cudaFree(db), "cudaFree b");
        CudaCheck.Runtime(CudaRuntime.cudaStreamDestroy(stream), "cudaStreamDestroy");
        Console.WriteLine(_failures == 0 ? "OK" : $"FAILED: {_failures} check(s)");
        return _failures == 0 ? 0 : 1;
    }

    private static void NvmlAndCublasLtSmoke()
    {
        nint lt;
        Expect(CublasLt.cublasLtCreate(&lt) == CublasLt.CUBLAS_STATUS_SUCCESS, "cublasLtCreate");
        Console.WriteLine($"cuBLASLt {CublasLt.cublasLtGetVersion()}, built against cudart {CublasLt.cublasLtGetCudartVersion()}");
        Expect(CublasLt.cublasLtDestroy(lt) == CublasLt.CUBLAS_STATUS_SUCCESS, "cublasLtDestroy");

        if (Nvml.nvmlInit() != Nvml.NVML_SUCCESS)
        {
            Console.WriteLine("NVML unavailable (nvmlInit failed); skipped");
            return;
        }

        try
        {
            var version = stackalloc byte[Nvml.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
            CudaCheck.Nvml(Nvml.nvmlSystemGetDriverVersion(version, Nvml.NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE), "nvmlSystemGetDriverVersion");
            Console.WriteLine($"NVML driver {Marshal.PtrToStringUTF8((nint)version)}");
            nvmlMemory_t mem;
            nint dev;
            CudaCheck.Nvml(Nvml.nvmlDeviceGetHandleByIndex(0, &dev), "nvmlDeviceGetHandleByIndex");
            CudaCheck.Nvml(Nvml.nvmlDeviceGetMemoryInfo(dev, &mem), "nvmlDeviceGetMemoryInfo");
            Console.WriteLine($"NVML device 0 memory: {mem.total >> 20} MiB total, {mem.free >> 20} MiB free");
            Expect(mem.total > 0 && mem.total == mem.free + mem.used, "nvmlMemory_t layout: total = free + used");
        }
        finally
        {
            Expect(Nvml.nvmlShutdown() == Nvml.NVML_SUCCESS, "nvmlShutdown");
        }
    }

    private static void Refused<T>(string what, Action act, Func<T, bool> check)
        where T : Exception
    {
        try
        {
            act();
            Expect(false, $"{what}: was not refused");
        }
        catch (T e)
        {
            Expect(check(e), $"{what}: refused with the wrong detail: {e.Message}");
            if (check(e))
            {
                Console.WriteLine($"refused as expected, {what}: {e.Message}");
            }
        }
    }

    private static void Expect(bool ok, string what)
    {
        if (!ok)
        {
            _failures++;
            Console.WriteLine($"FAIL: {what}");
        }
    }

    private static ushort ToBf16(float f)
    {
        var u = BitConverter.SingleToUInt32Bits(f);
        u += 0x7FFFu + ((u >> 16) & 1u);
        return (ushort)(u >> 16);
    }

    private static float FromBf16(ushort b) => BitConverter.UInt32BitsToSingle((uint)b << 16);
}
