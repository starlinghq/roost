using Starlings.Acquisition;
using Starlings.Roost.Abi;
using Starlings.Roost.Requests;

namespace Starlings.Roost.Kernels;

/// <summary>Where <see cref="KernelLibraryResolver"/> looks for the kernel library of an SM.</summary>
public sealed class KernelLibraryResolverOptions
{
    /// <summary>
    /// Directories with locally built or copied libraries, searched first (<see cref="KernelLibraryNames.Find"/>). When one
    /// holds the SM's file, nothing is downloaded.
    /// </summary>
    public IReadOnlyList<string> SearchDirectories { get; init; } = [];

    /// <summary>
    /// The release to fetch from when no directory holds the file; null for the release this assembly was built from
    /// (<see cref="KernelRelease.FromAssembly"/>), which a development build does not have.
    /// </summary>
    public KernelRelease? Release { get; init; }

    /// <summary>
    /// Cache root for fetched libraries, laid out as <c>&lt;root&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;commit&gt;/&lt;file&gt;</c>
    /// by <c>Starlings.Acquisition</c>; null for its default (<c>$STARLING_MODELS</c>, else the user cache directory).
    /// </summary>
    public string? CacheDirectory { get; init; }

    /// <summary>
    /// Whether a library missing from the cache may be downloaded. When false, only the directories and the cache are
    /// used (an offline host after a pre-fetch).
    /// </summary>
    public bool AllowDownload { get; init; } = true;

    /// <summary>The platform whose file names are used; null for the current one.</summary>
    public KernelPlatform? Platform { get; init; }

    /// <summary>The HTTP client for downloads (not disposed); null for the store's own client. Tests pass one with a fake handler.</summary>
    public HttpClient? HttpClient { get; init; }
}

/// <summary>
/// Finds the kernel library file for a device's SM (decision 12: Roost's kernel library is fetched at first run, pinned
/// by digest). A configured directory wins; otherwise the file is the release asset of the package version, fetched
/// through <c>Starlings.Acquisition</c>, verified by size and SHA-256 against the manifest before it is returned, and
/// cached. An SM the manifest does not admit is refused before any download. <see cref="KernelLibrary.Load"/> verifies the
/// digest again before loading.
/// </summary>
public sealed class KernelLibraryResolver
{
    private const string LockName = "roost-kernels";

    private readonly KernelLibraryResolverOptions _options;

    /// <summary>Creates a resolver. No network or file access happens here.</summary>
    /// <param name="options">The search directories, release and cache.</param>
    public KernelLibraryResolver(KernelLibraryResolverOptions? options = null)
    {
        _options = options ?? new KernelLibraryResolverOptions();
    }

    /// <summary>Resolves the library file for an SM, downloading it when allowed and needed.</summary>
    /// <param name="load">The manifest and SM (the same options <see cref="KernelLibrary.Load"/> takes).</param>
    /// <param name="progress">Download progress, or null.</param>
    /// <param name="cancellationToken">Cancels the download.</param>
    /// <returns>The absolute path of a file whose digest matches the manifest.</returns>
    /// <exception cref="ProfileUnsupportedException">
    /// The manifest does not admit the SM, builds no library for it, or no file is configured and this build records no release.
    /// </exception>
    /// <exception cref="KernelLoadException">
    /// The fetched or cached bytes differ from the manifest, or the release has no asset for the SM
    /// (<see cref="RoostErrorCode.AssetMismatch"/>); a network failure after retries (<see cref="RoostErrorCode.Internal"/>).
    /// </exception>
    public async Task<string> ResolveAsync(KernelLoadOptions load, IProgress<ModelAssetProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        var (assetLock, fileName, local) = Prepare(load);
        if (local is not null)
        {
            return local;
        }

        using var store = new ModelAssetStore(StoreOptions(), _options.HttpClient);
        try
        {
            var snapshot = _options.AllowDownload
                ? await store.DownloadAsync(assetLock!, progress, cancellationToken).ConfigureAwait(false)
                : store.OpenLocal(assetLock!);
            return snapshot.GetFilePath(fileName);
        }
        catch (ModelAcquisitionException e)
        {
            throw Translate(e, load.Sm, fileName);
        }
    }

    /// <summary>Resolves the library file for an SM from the directories and the cache only; never downloads.</summary>
    /// <param name="load">The manifest and SM.</param>
    /// <returns>The absolute path of a file whose digest matches the manifest.</returns>
    /// <exception cref="ProfileUnsupportedException">As for <see cref="ResolveAsync"/>.</exception>
    /// <exception cref="KernelLoadException">
    /// The cached bytes differ from the manifest (<see cref="RoostErrorCode.AssetMismatch"/>), or the file is not cached
    /// (<see cref="ProfileUnsupportedException"/>).
    /// </exception>
    public string ResolveLocal(KernelLoadOptions load)
    {
        var (assetLock, fileName, local) = Prepare(load);
        if (local is not null)
        {
            return local;
        }

        using var store = new ModelAssetStore(StoreOptions(), _options.HttpClient);
        try
        {
            return store.OpenLocal(assetLock!).GetFilePath(fileName);
        }
        catch (ModelAcquisitionException e)
        {
            throw Translate(e, load.Sm, fileName);
        }
    }

    /// <summary>
    /// Fetches and verifies the libraries of several SMs into the cache ahead of time (the pre-fetch of decision 12), so a
    /// later <see cref="ResolveLocal"/> or a resolver with <see cref="KernelLibraryResolverOptions.AllowDownload"/> off
    /// finds them offline.
    /// </summary>
    /// <param name="manifest">The manifest.</param>
    /// <param name="sms">The SMs to fetch.</param>
    /// <param name="allowUntested">Fetch SMs whose manifest status is untested too.</param>
    /// <param name="cancellationToken">Cancels the downloads.</param>
    /// <returns>The resolved path per SM.</returns>
    public async Task<IReadOnlyDictionary<int, string>> PrefetchAsync(
        KernelManifest manifest,
        IEnumerable<int> sms,
        bool allowUntested = false,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentNullException.ThrowIfNull(sms);
        var paths = new Dictionary<int, string>();
        foreach (var sm in sms)
        {
            paths[sm] = await ResolveAsync(new KernelLoadOptions(manifest, sm, allowUntested), null, cancellationToken).ConfigureAwait(false);
        }

        return paths;
    }

    /// <summary>The asset lock for one SM's library: what is fetched, from where, and the size and digest it must have.</summary>
    /// <param name="load">The manifest and SM.</param>
    /// <param name="release">The release.</param>
    /// <param name="platform">The platform whose file name is used.</param>
    /// <returns>The lock.</returns>
    /// <exception cref="ProfileUnsupportedException">The manifest does not admit the SM or builds no library for it.</exception>
    /// <exception cref="KernelLoadException">The manifest records no size for the SM's library (<see cref="RoostErrorCode.AssetMismatch"/>).</exception>
    public static ModelAssetLock LockFor(KernelLoadOptions load, KernelRelease release, KernelPlatform platform)
    {
        ArgumentNullException.ThrowIfNull(load);
        ArgumentNullException.ThrowIfNull(release);
        KernelLibrary.Admit(load);
        var expected = KernelLibrary.ExpectedBinary(load);
        if (expected.Bytes <= 0)
        {
            throw new KernelLoadException(
                RoostErrorCode.AssetMismatch,
                $"the manifest records no size for the SM {load.Sm} library; a released manifest records size and digest");
        }

        var fileName = KernelLibraryNames.FileName(load.Sm, platform);
        return new ModelAssetLock(
            LockName,
            release.Repository,
            release.Commit,
            [new ModelAssetFile(fileName, expected.Bytes, expected.Sha256, release.AssetUri(fileName))]);
    }

    private (ModelAssetLock? Lock, string FileName, string? Local) Prepare(KernelLoadOptions load)
    {
        ArgumentNullException.ThrowIfNull(load);

        // Refuse an SM the manifest does not admit before touching the disk or the network.
        KernelLibrary.Admit(load);
        _ = KernelLibrary.ExpectedBinary(load);

        var platform = _options.Platform ?? KernelLibraryNames.Current();
        var fileName = KernelLibraryNames.FileName(load.Sm, platform);
        var local = KernelLibraryNames.Find(load.Sm, _options.SearchDirectories, platform);
        if (local is not null)
        {
            return (null, fileName, local);
        }

        var release = _options.Release ?? KernelRelease.FromAssembly()
            ?? throw new ProfileUnsupportedException(
                OpId.None,
                load.Sm,
                $"no directory holds {fileName} and this build records no release to fetch it from (a development build); " +
                "build the kernel library and add its directory to the search directories");
        return (LockFor(load, release, platform), fileName, null);
    }

    private ModelAcquisitionOptions StoreOptions() => new()
    {
        CacheDirectory = _options.CacheDirectory,
        Verification = CacheVerification.Sha256,

        // The assets are public; no Hugging Face token is ever read or sent for them.
        UseEnvironmentToken = false,
    };

    private static KernelLoadException Translate(ModelAcquisitionException e, int sm, string fileName) => e switch
    {
        ModelAssetIntegrityException integrity => new KernelLoadException(
            RoostErrorCode.AssetMismatch,
            $"kernel library {fileName} for SM {sm} does not match the manifest: expected {integrity.ExpectedSize} bytes, " +
            $"sha256 {integrity.ExpectedSha256}; got {integrity.ActualSize} bytes, sha256 {integrity.ActualSha256 ?? "(not hashed)"}",
            e),
        ModelAssetNotFoundException => new KernelLoadException(
            RoostErrorCode.AssetMismatch,
            $"the release has no asset {fileName} for SM {sm}, though the manifest records its digest: the package and the release disagree",
            e),
        ModelAssetMissingException => new ProfileUnsupportedException(
            OpId.None,
            sm,
            $"kernel library {fileName} is not in the cache and downloading is off; pre-fetch it first"),
        _ => new KernelLoadException(RoostErrorCode.Internal, $"could not fetch kernel library {fileName} for SM {sm}: {e.Message}", e),
    };
}
