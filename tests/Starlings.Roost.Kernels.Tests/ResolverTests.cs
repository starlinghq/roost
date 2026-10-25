using System.Net;
using System.Security.Cryptography;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels.Tests;

/// <summary>
/// Kernel library acquisition by digest (decision 12, R-060): a fake HTTP handler stands in for the release, so no test
/// touches the network.
/// </summary>
public sealed class ResolverTests : IDisposable
{
    private const string Commit = "0123456789abcdef0123456789abcdef01234567";
    private static readonly byte[] Library = [.. Enumerable.Range(0, 4096).Select(i => (byte)(i * 7))];

    private readonly string _root = Directory.CreateTempSubdirectory("roost-resolver-").FullName;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private static string Sha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));

    private static KernelLoadOptions Options(byte[]? recorded = null, int sm = 86, long? bytes = null, string sm86Status = "experimental")
    {
        recorded ??= Library;
        return new KernelLoadOptions(TestManifest.Build(sm86Status, Sha(recorded), bytes ?? recorded.Length), sm);
    }

    private static KernelRelease Release() => new("0.1.0-preview.1", Commit);

    private (KernelLibraryResolver Resolver, FakeRelease Server) Resolver(byte[]? served = null, bool allowDownload = true, IReadOnlyList<string>? dirs = null)
    {
        var server = new FakeRelease(served ?? Library);
        var resolver = new KernelLibraryResolver(new KernelLibraryResolverOptions
        {
            Release = Release(),
            CacheDirectory = Path.Combine(_root, "cache"),
            HttpClient = new HttpClient(server),
            Platform = KernelPlatform.Linux,
            AllowDownload = allowDownload,
            SearchDirectories = dirs ?? [],
        });
        return (resolver, server);
    }

    [Fact]
    public async Task FetchesTheReleaseAssetVerifiesItAndCachesIt()
    {
        var (resolver, server) = Resolver();
        var path = await resolver.ResolveAsync(Options(), cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(Library, await File.ReadAllBytesAsync(path, TestContext.Current.CancellationToken));
        Assert.Equal(Path.Combine(_root, "cache", "starlinghq", "roost", Commit, "libroost_kernels_sm86.so"), path);
        var request = Assert.Single(server.Requests);
        Assert.Equal("https://github.com/starlinghq/roost/releases/download/v0.1.0-preview.1/libroost_kernels_sm86.so", request.AbsoluteUri);

        // A second resolve reads the verified cache and makes no request; so does an offline resolve.
        Assert.Equal(path, await resolver.ResolveAsync(Options(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(path, resolver.ResolveLocal(Options()));
        Assert.Single(server.Requests);
    }

    [Fact]
    public async Task BytesThatDifferFromTheManifestAreAnAssetMismatch()
    {
        var other = Library.Select(b => (byte)(b ^ 1)).ToArray();
        var (resolver, _) = Resolver(served: other);
        var e = await Assert.ThrowsAsync<KernelLoadException>(
            () => resolver.ResolveAsync(Options(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("does not match the manifest", e.Message);
        Assert.False(File.Exists(Path.Combine(_root, "cache", "starlinghq", "roost", Commit, "libroost_kernels_sm86.so")));
    }

    [Fact]
    public async Task ACorruptedCacheEntryIsRefusedOffline()
    {
        var (resolver, _) = Resolver();
        var path = await resolver.ResolveAsync(Options(), cancellationToken: TestContext.Current.CancellationToken);
        var bytes = await File.ReadAllBytesAsync(path, TestContext.Current.CancellationToken);
        bytes[100] ^= 0xFF;
        await File.WriteAllBytesAsync(path, bytes, TestContext.Current.CancellationToken);

        var e = Assert.ThrowsAny<KernelLoadException>(() => resolver.ResolveLocal(Options()));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
    }

    [Fact]
    public async Task AnSmTheManifestDoesNotBuildIsRefusedWithoutARequest()
    {
        var (resolver, server) = Resolver();
        var e = await Assert.ThrowsAsync<ProfileUnsupportedException>(
            () => resolver.ResolveAsync(Options(sm: 120), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(RoostErrorCode.ProfileUnsupported, e.Error.Code);
        Assert.Equal(120, e.Sm);
        Assert.Empty(server.Requests);
    }

    [Theory]
    [InlineData(80)] // untested, no opt-in
    [InlineData(90)] // unsupported
    public async Task AnSmTheManifestDoesNotAdmitIsRefusedWithoutARequest(int sm)
    {
        var (resolver, server) = Resolver();
        await Assert.ThrowsAsync<ProfileUnsupportedException>(
            () => resolver.ResolveAsync(Options(sm: sm), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task AMissingReleaseAssetIsAnAssetMismatch()
    {
        var (resolver, server) = Resolver();
        server.Status = HttpStatusCode.NotFound;
        var e = await Assert.ThrowsAsync<KernelLoadException>(
            () => resolver.ResolveAsync(Options(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Contains("release has no asset", e.Message);
    }

    [Fact]
    public async Task AManifestWithoutASizeIsRefusedBeforeARequest()
    {
        var (resolver, server) = Resolver();
        var e = await Assert.ThrowsAsync<KernelLoadException>(
            () => resolver.ResolveAsync(Options(bytes: 0), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Equal(RoostErrorCode.AssetMismatch, e.Error.Code);
        Assert.Empty(server.Requests);
    }

    [Fact]
    public void OfflineWithNothingCachedIsProfileUnsupported()
    {
        var (resolver, server) = Resolver(allowDownload: false);
        var e = Assert.Throws<ProfileUnsupportedException>(() => resolver.ResolveLocal(Options()));
        Assert.Contains("pre-fetch", e.Message);
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task ALocalDirectoryWinsAndNothingIsFetched()
    {
        var dir = Directory.CreateDirectory(Path.Combine(_root, "local")).FullName;
        var file = Path.Combine(dir, "libroost_kernels_sm86.so");
        await File.WriteAllBytesAsync(file, Library, TestContext.Current.CancellationToken);
        var (resolver, server) = Resolver(dirs: [dir]);

        Assert.Equal(file, await resolver.ResolveAsync(Options(), cancellationToken: TestContext.Current.CancellationToken));
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task PrefetchFillsTheCacheForAnOfflineResolve()
    {
        var (resolver, server) = Resolver();
        var paths = await resolver.PrefetchAsync(Options().Manifest, [86], cancellationToken: TestContext.Current.CancellationToken);
        Assert.Single(server.Requests);

        var offline = new KernelLibraryResolver(new KernelLibraryResolverOptions
        {
            Release = Release(),
            CacheDirectory = Path.Combine(_root, "cache"),
            Platform = KernelPlatform.Linux,
            AllowDownload = false,
        });
        Assert.Equal(paths[86], offline.ResolveLocal(Options()));
    }

    [Fact]
    public void ADevelopmentBuildHasNoReleaseAndIsRefused()
    {
        Assert.Null(KernelRelease.FromInformationalVersion("0.0.0-dev+" + Commit));
        Assert.Null(KernelRelease.FromInformationalVersion("0.1.0-preview.1"));
        Assert.Null(KernelRelease.FromInformationalVersion("0.1.0+abc"));
        var release = KernelRelease.FromInformationalVersion("0.1.0-preview.1+" + Commit.ToUpperInvariant());
        Assert.NotNull(release);
        Assert.Equal(("0.1.0-preview.1", Commit), (release.Version, release.Commit));

        // The test assembly's Kernels build is a development build (VersionPrefix 0.0.0), so with no release configured
        // and no directory, the resolver refuses with a typed error instead of guessing a URL.
        var resolver = new KernelLibraryResolver(new KernelLibraryResolverOptions { CacheDirectory = _root, Platform = KernelPlatform.Linux });
        var e = Assert.Throws<ProfileUnsupportedException>(() => resolver.ResolveLocal(Options()));
        Assert.Contains("records no release", e.Message);
    }

    [Fact]
    public void AReleaseRefusesAShortCommitOrPlainHttp()
    {
        Assert.Throws<ArgumentException>(() => new KernelRelease("0.1.0", "abc123"));
        Assert.Throws<ArgumentException>(() => new KernelRelease("0.1.0", Commit, new Uri("http://example.test/")));
        Assert.Equal(
            "https://mirror.example.test/roost/libroost_kernels_sm86.so",
            new KernelRelease("0.1.0", Commit, new Uri("https://mirror.example.test/roost")).AssetUri("libroost_kernels_sm86.so").AbsoluteUri);
    }

    [Fact]
    public void TheAssemblyEmbedsTheRepositoryManifest()
    {
        var embedded = KernelManifest.Embedded();
        Assert.Equal(TestManifest.Repository().Sha256, embedded.Sha256);
    }

    /// <summary>Answers every GET with the served bytes (or a status), and records the requested addresses.</summary>
    private sealed class FakeRelease(byte[] served) : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = [];

        public HttpStatusCode Status { get; set; } = HttpStatusCode.OK;

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request.RequestUri!);
            var response = new HttpResponseMessage(Status) { RequestMessage = request };
            if (Status == HttpStatusCode.OK)
            {
                response.Content = new ByteArrayContent(served);
                response.Content.Headers.ContentLength = served.Length;
            }

            return Task.FromResult(response);
        }
    }
}
