using Starlings.Roost.Cuda;
using Starlings.Roost.Engine.Admission;
using Starlings.Roost.Worker.Memory;

namespace Starlings.Roost.Worker.Tests;

/// <summary>The NVML memory reader (R-044) over a fake NVML: every failure is a typed reading, and it recovers.</summary>
public sealed class NvmlDeviceMemoryReaderTests
{
    [Fact]
    public void ReadsTotalAndFreeThroughTheV2Call()
    {
        var nvml = new FakeNvml { Total = 24_000, Free = 10_000 };
        using var reader = NvmlDeviceMemoryReader.ForIndex(1, nvml);

        var reading = reader.Read();
        Assert.Equal(DeviceMemoryReadStatus.Read, reading.Status);
        Assert.Equal(new DeviceMemory(24_000, 10_000), reading.Memory);
        Assert.Equal(1u, nvml.LastIndex);
        Assert.Equal(1, nvml.InitCalls);

        nvml.Free = 3_000; // a co-tenant allocates
        Assert.Equal(3_000, reader.Read().Memory.FreeBytes);
        Assert.Equal(1, nvml.InitCalls); // initialised once, handle kept
        Assert.Equal(1, nvml.HandleCalls);
        Assert.Equal(2, nvml.V2Calls);
        Assert.Equal(0, nvml.V1Calls);
    }

    [Fact]
    public void ByUuidLooksTheDeviceUpByUuid()
    {
        var nvml = new FakeNvml { Total = 8, Free = 4 };
        using var reader = NvmlDeviceMemoryReader.ForUuid("GPU-0000", nvml);
        Assert.True(reader.Read().IsReadable);
        Assert.Equal("GPU-0000", nvml.LastUuid);
        Assert.Throws<ArgumentException>(() => NvmlDeviceMemoryReader.ForUuid(" ", nvml));
    }

    [Theory]
    [InlineData(NvmlReturn.FunctionNotFound)]
    [InlineData(NvmlReturn.NotSupported)]
    [InlineData(NvmlReturn.ArgumentVersionMismatch)]
    public void FallsBackToTheV1CallOnADriverWithoutV2(int v2Status)
    {
        var nvml = new FakeNvml { Total = 24_000, Free = 10_000, V2Status = v2Status };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        Assert.Equal(10_000, reader.Read().Memory.FreeBytes);
        Assert.Equal(10_000, reader.Read().Memory.FreeBytes);
        Assert.Equal(1, nvml.V2Calls); // v2 is not tried again
        Assert.Equal(2, nvml.V1Calls);
    }

    [Fact]
    public void AMissingLibraryIsTypedAndNotProbedAgain()
    {
        var nvml = new FakeNvml { InitStatus = NvmlReturn.LibraryNotFound };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        Assert.Equal(DeviceMemoryReadStatus.LibraryMissing, reader.Read().Status);
        Assert.Equal(DeviceMemoryReadStatus.LibraryMissing, reader.Read().Status);
        Assert.Equal(1, nvml.InitCalls);
    }

    [Fact]
    public void AnInitFailureForAnotherReasonIsRetried()
    {
        var nvml = new FakeNvml { Total = 8, Free = 4, InitStatus = NvmlReturn.DriverNotLoaded };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        var first = reader.Read();
        Assert.Equal(DeviceMemoryReadStatus.LibraryMissing, first.Status);
        Assert.Contains("nvmlInit", first.Detail, StringComparison.Ordinal);

        nvml.InitStatus = NvmlReturn.Success;
        Assert.True(reader.Read().IsReadable);
        Assert.Equal(2, nvml.InitCalls);
    }

    [Fact]
    public void ALostDeviceIsTypedAndItsHandleIsLookedUpAgainWhenItReturns()
    {
        var nvml = new FakeNvml { Total = 24_000, Free = 10_000 };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        Assert.True(reader.Read().IsReadable);

        nvml.V2Status = NvmlReturn.GpuIsLost;
        var lost = reader.Read();
        Assert.Equal(DeviceMemoryReadStatus.DeviceLost, lost.Status);
        Assert.Contains("nvmlDeviceGetMemoryInfo_v2", lost.Detail, StringComparison.Ordinal);

        nvml.HandleStatus = NvmlReturn.GpuIsLost;
        Assert.Equal(DeviceMemoryReadStatus.DeviceLost, reader.Read().Status);

        nvml.HandleStatus = NvmlReturn.Success;
        nvml.V2Status = NvmlReturn.Success;
        Assert.True(reader.Read().IsReadable);
        Assert.Equal(3, nvml.HandleCalls);
    }

    [Fact]
    public void AnUnknownDeviceIsTyped()
    {
        var nvml = new FakeNvml { HandleStatus = NvmlReturn.NotFound };
        using var reader = NvmlDeviceMemoryReader.ForIndex(7, nvml);
        Assert.Equal(DeviceMemoryReadStatus.DeviceNotFound, reader.Read().Status);
    }

    [Fact]
    public void AnUninitialisedLibraryIsInitialisedAgain()
    {
        var nvml = new FakeNvml { Total = 8, Free = 4 };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        Assert.True(reader.Read().IsReadable);
        nvml.V2Status = NvmlReturn.Uninitialized;
        Assert.Equal(DeviceMemoryReadStatus.Failed, reader.Read().Status);
        nvml.V2Status = NvmlReturn.Success;
        Assert.True(reader.Read().IsReadable);
        Assert.Equal(2, nvml.InitCalls);
        Assert.Equal(2, nvml.HandleCalls);
    }

    [Fact]
    public void AnImpossibleReadingIsFailed()
    {
        var nvml = new FakeNvml { Total = 8, Free = 9 };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        Assert.Equal(DeviceMemoryReadStatus.Failed, reader.Read().Status);
    }

    [Fact]
    public void DisposeShutsDownOnlyWhatItInitialised()
    {
        var unused = new FakeNvml();
        NvmlDeviceMemoryReader.ForIndex(0, unused).Dispose();
        Assert.Equal(0, unused.ShutdownCalls);

        var nvml = new FakeNvml { Total = 8, Free = 4 };
        var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        Assert.True(reader.Read().IsReadable);
        reader.Dispose();
        reader.Dispose();
        Assert.Equal(1, nvml.ShutdownCalls);
        Assert.Equal(DeviceMemoryReadStatus.Failed, reader.Read().Status);
    }

    [Theory]
    [InlineData(NvmlReturn.LibraryNotFound, DeviceMemoryReadStatus.LibraryMissing)]
    [InlineData(NvmlReturn.DriverNotLoaded, DeviceMemoryReadStatus.LibraryMissing)]
    [InlineData(NvmlReturn.LibRmVersionMismatch, DeviceMemoryReadStatus.LibraryMissing)]
    [InlineData(NvmlReturn.NoPermission, DeviceMemoryReadStatus.LibraryMissing)]
    [InlineData(NvmlReturn.GpuIsLost, DeviceMemoryReadStatus.DeviceLost)]
    [InlineData(NvmlReturn.Unknown, DeviceMemoryReadStatus.Failed)]
    public void InitStatusesAreClassified(int status, DeviceMemoryReadStatus expected) =>
        Assert.Equal(expected, NvmlDeviceMemoryReader.ClassifyInit(status));

    [Theory]
    [InlineData(NvmlReturn.GpuIsLost, DeviceMemoryReadStatus.DeviceLost)]
    [InlineData(NvmlReturn.ResetRequired, DeviceMemoryReadStatus.DeviceLost)]
    [InlineData(NvmlReturn.NotFound, DeviceMemoryReadStatus.DeviceNotFound)]
    [InlineData(NvmlReturn.InvalidArgument, DeviceMemoryReadStatus.DeviceNotFound)]
    [InlineData(NvmlReturn.LibraryNotFound, DeviceMemoryReadStatus.LibraryMissing)]
    [InlineData(NvmlReturn.Unknown, DeviceMemoryReadStatus.Failed)]
    public void DeviceStatusesAreClassified(int status, DeviceMemoryReadStatus expected) =>
        Assert.Equal(expected, NvmlDeviceMemoryReader.ClassifyDevice(status));

    [Fact]
    public void TheRealLibraryNeverThrowsWhetherOrNotADriverIsInstalled()
    {
        // CI hosts have no NVIDIA driver: the reading is typed unreadable. A GPU host reads its device 0.
        using var reader = NvmlDeviceMemoryReader.ForIndex(0);
        var reading = reader.Read();
        if (reading.IsReadable)
        {
            Assert.InRange(reading.Memory.FreeBytes, 0, reading.Memory.TotalBytes);
        }
        else
        {
            Assert.NotEqual(DeviceMemoryReadStatus.Read, reading.Status);
            Assert.NotNull(reading.Detail);
        }
    }

    [Fact]
    public void ThePolledSourceOverTheReaderNeverThrowsIntoAdmission()
    {
        var nvml = new FakeNvml { InitStatus = NvmlReturn.LibraryNotFound };
        using var reader = NvmlDeviceMemoryReader.ForIndex(0, nvml);
        using var source = new PolledDeviceMemorySource(reader, new PolledDeviceMemorySourceOptions { Period = TimeSpan.FromHours(1) });
        // Refresh answers false while the timer's first read runs, and Reads counts it only when it ends.
        Assert.True(SpinWait.SpinUntil(() => source.Refresh() || source.Reads > 0, TimeSpan.FromSeconds(30)));
        Assert.False(source.TryRead(out _));
        Assert.Equal(DeviceMemoryReadStatus.LibraryMissing, source.Current.Status);
    }

    private sealed class FakeNvml : INvmlApi
    {
        public ulong Total { get; set; }

        public ulong Free { get; set; }

        public int InitStatus { get; set; }

        public int HandleStatus { get; set; }

        public int V2Status { get; set; }

        public int InitCalls { get; private set; }

        public int ShutdownCalls { get; private set; }

        public int HandleCalls { get; private set; }

        public int V2Calls { get; private set; }

        public int V1Calls { get; private set; }

        public uint? LastIndex { get; private set; }

        public string? LastUuid { get; private set; }

        public int Init()
        {
            InitCalls++;
            return InitStatus;
        }

        public int Shutdown()
        {
            ShutdownCalls++;
            return NvmlReturn.Success;
        }

        public int DeviceGetHandleByIndex(uint index, out nint device)
        {
            HandleCalls++;
            LastIndex = index;
            device = HandleStatus == NvmlReturn.Success ? 0x1000 : 0;
            return HandleStatus;
        }

        public int DeviceGetHandleByUuid(string uuid, out nint device)
        {
            HandleCalls++;
            LastUuid = uuid;
            device = HandleStatus == NvmlReturn.Success ? 0x1000 : 0;
            return HandleStatus;
        }

        public int DeviceGetMemoryInfoV2(nint device, out ulong totalBytes, out ulong freeBytes)
        {
            V2Calls++;
            Assert.Equal(0x1000, device);
            (totalBytes, freeBytes) = V2Status == NvmlReturn.Success ? (Total, Free) : (0UL, 0UL);
            return V2Status;
        }

        public int DeviceGetMemoryInfo(nint device, out ulong totalBytes, out ulong freeBytes)
        {
            V1Calls++;
            Assert.Equal(0x1000, device);
            (totalBytes, freeBytes) = (Total, Free);
            return NvmlReturn.Success;
        }

        public string Describe(int status) => $"status {status}";
    }
}
