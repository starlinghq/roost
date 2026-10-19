using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Starlings.Roost.Abi;
using Starlings.Roost.Plans;
using Starlings.Roost.Transport;

namespace Starlings.Roost.WorkerHost.Tests.Cuda;

/// <summary>
/// The S8 step as an <see cref="IStepExecutor"/>: 4 KiB plan to the device, a kernel that spins for the time the plan
/// asks for, 1 KiB result back, with copies straight from and to ring memory (the segment is registered). The native
/// side is <c>native/roost_spin.cu</c>. Classification follows S8 condition 2: a failed call is probed with a memset and
/// a sync, and a second failure is sticky.
/// </summary>
internal static unsafe partial class CudaSpinExecutor
{
    public const int PlanBytes = 4096;
    public const int ResultBytes = 4096;
    public const int BodyBytes = 1024;

    static CudaSpinExecutor()
    {
        NativeLibrary.SetDllImportResolver(typeof(CudaSpinExecutor).Assembly, (name, assembly, path) =>
        {
            if (name != "roostspin")
            {
                return 0;
            }

            var file = Environment.GetEnvironmentVariable("ROOST_SPIN_LIB")
                ?? throw new InvalidOperationException("Set ROOST_SPIN_LIB to the path of libroostspin.so (native/build.sh).");
            return NativeLibrary.Load(file);
        });
    }

    /// <summary>Offset of the 8-byte spin time in the plan buffer, after the header.</summary>
    public static int SpinOffset => (sizeof(StepPlanHeader) + 7) & ~7;

    public static ImmutableArray<byte> ManifestDigest { get; } = Digest(0x4D);

    public static ImmutableArray<byte> WeightsDigest { get; } = Digest(0x57);

    public static IStepExecutor Create() => new Executor();

    /// <summary>Writes a plan whose step spins for <paramref name="gpuNs"/>.</summary>
    public static void FillPlan(Span<byte> buffer, ulong stepId, ulong gpuNs)
    {
        var header = new StepPlanHeader
        {
            Magic = 0x4C505352u,
            LayoutVersion = 1,
            HeaderSize = (uint)sizeof(StepPlanHeader),
            StepId = stepId,
            TotalBytes = (ulong)(SpinOffset + 8),
        };
        Unsafe.WriteUnaligned(ref buffer[0], header);
        Unsafe.WriteUnaligned(ref buffer[SpinOffset], gpuNs);
    }

    /// <summary>Opens the device in this process and registers the given host memory; used by the direct baseline.</summary>
    public static int OpenDirect(int ordinal, nint address, long bytes)
    {
        Span<byte> uuid = stackalloc byte[16];
        fixed (byte* p = uuid)
        {
            var error = Open(ordinal, p);
            if (error != 0)
            {
                return error;
            }
        }

        return Register(address, (nuint)bytes);
    }

    /// <summary>Registers more host memory with the device.</summary>
    public static int RegisterHost(nint address, long bytes) => Register(address, (nuint)bytes);

    /// <summary>One step with no ring in between; returns the CUDA status.</summary>
    public static int StepDirect(byte* plan, byte* body, out ulong eventNs)
    {
        ulong ns;
        var error = Step(plan, (nuint)SpinOffset, body, &ns);
        eventNs = ns;
        return error;
    }

    [LibraryImport("roostspin", EntryPoint = "roost_spin_open")]
    private static partial int Open(int ordinal, byte* uuid16);

    [LibraryImport("roostspin", EntryPoint = "roost_spin_register")]
    private static partial int Register(nint address, nuint bytes);

    [LibraryImport("roostspin", EntryPoint = "roost_spin_step")]
    private static partial int Step(byte* plan, nuint nsOffset, byte* body, ulong* eventNs);

    [LibraryImport("roostspin", EntryPoint = "roost_spin_probe")]
    private static partial int Probe();

    private static ImmutableArray<byte> Digest(byte seed)
    {
        var bytes = new byte[32];
        for (var i = 0; i < bytes.Length; i++)
        {
            bytes[i] = (byte)(seed + i);
        }

        return ImmutableCollectionsMarshal.AsImmutableArray(bytes);
    }

    private sealed class Executor : IStepExecutor
    {
        public WorkerIdentity Identity => new(KernelAbi.Version, ManifestDigest, WeightsDigest);

        public HandshakeReject OpenDevice(int deviceOrdinal, Span<byte> deviceUuid)
        {
            fixed (byte* p = deviceUuid)
            {
                return Open(deviceOrdinal, p) == 0 ? HandshakeReject.None : HandshakeReject.Device;
            }
        }

        public void RegisterSegment(nint address, long bytes)
        {
            var error = Register(address, (nuint)bytes);
            if (error != 0)
            {
                throw new InvalidOperationException($"cudaHostRegister failed: {error}");
            }
        }

        public HandshakeReject LoadWeights() => HandshakeReject.None;

        public StepOutcome Execute(ReadOnlySpan<byte> plan, Span<byte> result)
        {
            var header = new StepResultHeader
            {
                Magic = 0x53525352u,
                LayoutVersion = 1,
                HeaderSize = (uint)sizeof(StepResultHeader),
                StepId = Unsafe.ReadUnaligned<StepPlanHeader>(ref Unsafe.AsRef(in plan[0])).StepId,
                TotalBytes = (ulong)(sizeof(StepResultHeader) + BodyBytes),
            };
            fixed (byte* p = plan)
            fixed (byte* r = result)
            {
                ulong ns;
                var error = Step(p, (nuint)SpinOffset, r + sizeof(StepResultHeader), &ns);
                if (error != 0)
                {
                    return Probe() == 0 ? StepOutcome.Recoverable(error) : StepOutcome.Sticky(error);
                }
            }

            Unsafe.WriteUnaligned(ref result[0], header);
            return StepOutcome.Ok(sizeof(StepResultHeader) + BodyBytes);
        }

        public void Dispose()
        {
        }
    }
}
