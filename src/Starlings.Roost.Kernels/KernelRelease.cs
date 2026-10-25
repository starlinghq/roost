using System.Globalization;
using System.Reflection;
using System.Text.RegularExpressions;

namespace Starlings.Roost.Kernels;

/// <summary>
/// The Roost release a package version was built from: where its kernel library assets live (decision 12, R-060). The
/// release workflow publishes one <c>libroost_kernels_sm&lt;N&gt;.so</c> per SM as assets of the GitHub release
/// <c>v&lt;version&gt;</c>, and the package embeds the manifest whose digests those assets must match.
/// </summary>
public sealed partial record KernelRelease
{
    /// <summary>The repository the release assets are published from, <c>owner/name</c>.</summary>
    public const string DefaultRepository = "starlinghq/roost";

    /// <summary>Creates a release description.</summary>
    /// <param name="version">The package version without a leading <c>v</c>, for example <c>0.1.0-preview.1</c>.</param>
    /// <param name="commit">The full 40-character commit SHA the release was built from.</param>
    /// <param name="assetBaseAddress">
    /// The address the asset file names are appended to; null for
    /// <c>https://github.com/starlinghq/roost/releases/download/v&lt;version&gt;/</c>.
    /// </param>
    /// <param name="repository">The repository, <c>owner/name</c>; it names the cache directory.</param>
    /// <exception cref="ArgumentException">The version is empty, the commit is not 40 hex characters, or the address is not absolute https.</exception>
    public KernelRelease(string version, string commit, Uri? assetBaseAddress = null, string repository = DefaultRepository)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(version);
        ArgumentException.ThrowIfNullOrWhiteSpace(repository);
        if (!CommitPattern().IsMatch(commit ?? string.Empty))
        {
            throw new ArgumentException($"the commit must be a full 40-character hex SHA, not '{commit}'", nameof(commit));
        }

        var address = assetBaseAddress ?? new Uri($"https://github.com/{repository}/releases/download/v{version}/");
        if (!address.IsAbsoluteUri || address.Scheme != Uri.UriSchemeHttps)
        {
            throw new ArgumentException($"the asset address must be absolute https, not '{address}'", nameof(assetBaseAddress));
        }

        Version = version;
        Commit = commit!.ToLowerInvariant();
        Repository = repository;
        AssetBaseAddress = address.AbsoluteUri.EndsWith('/') ? address : new Uri(address.AbsoluteUri + "/");
    }

    /// <summary>The package version, for example <c>0.1.0-preview.1</c>.</summary>
    public string Version { get; }

    /// <summary>The commit SHA, lower-case hex.</summary>
    public string Commit { get; }

    /// <summary>The repository, <c>owner/name</c>.</summary>
    public string Repository { get; }

    /// <summary>The address asset file names are resolved against; ends with a slash.</summary>
    public Uri AssetBaseAddress { get; }

    /// <summary>The download address of one asset.</summary>
    /// <param name="fileName">The asset file name, for example <c>libroost_kernels_sm86.so</c>.</param>
    /// <returns>The absolute address.</returns>
    public Uri AssetUri(string fileName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(fileName);
        return new Uri(AssetBaseAddress, Uri.EscapeDataString(fileName));
    }

    /// <summary>
    /// The release an assembly was built from, read from its informational version <c>&lt;version&gt;+&lt;commit&gt;</c>
    /// (the SDK appends the source revision when building from a Git checkout). A development build (version suffix
    /// <c>dev</c>, prefix <c>0.0.0</c>) or a version without a full commit SHA has no release: null.
    /// </summary>
    /// <param name="assembly">The assembly; null for <c>Starlings.Roost.Kernels</c>.</param>
    /// <returns>The release, or null.</returns>
    public static KernelRelease? FromAssembly(Assembly? assembly = null)
    {
        var informational = (assembly ?? typeof(KernelRelease).Assembly)
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        return FromInformationalVersion(informational);
    }

    /// <summary>Parses an informational version <c>&lt;version&gt;+&lt;commit&gt;</c>; see <see cref="FromAssembly"/>.</summary>
    /// <param name="informationalVersion">The value, or null.</param>
    /// <returns>The release, or null for a development build or a value without a full commit SHA.</returns>
    public static KernelRelease? FromInformationalVersion(string? informationalVersion)
    {
        if (string.IsNullOrWhiteSpace(informationalVersion))
        {
            return null;
        }

        var plus = informationalVersion.IndexOf('+', StringComparison.Ordinal);
        if (plus <= 0)
        {
            return null;
        }

        var version = informationalVersion[..plus];
        var commit = informationalVersion[(plus + 1)..];
        var dot = commit.IndexOf('.', StringComparison.Ordinal);
        if (dot >= 0)
        {
            commit = commit[..dot];
        }

        if (!CommitPattern().IsMatch(commit) || IsDevelopmentVersion(version))
        {
            return null;
        }

        return new KernelRelease(version, commit);
    }

    private static bool IsDevelopmentVersion(string version) =>
        version.StartsWith("0.0.0", StringComparison.Ordinal)
        || version.EndsWith("-dev", StringComparison.Ordinal)
        || version.Contains("-dev.", StringComparison.Ordinal);

    /// <inheritdoc />
    public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{Repository} v{Version} ({Commit})");

    [GeneratedRegex("^[0-9a-fA-F]{40}$")]
    private static partial Regex CommitPattern();
}
