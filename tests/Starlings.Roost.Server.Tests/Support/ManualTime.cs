namespace Starlings.Roost.Server.Tests.Support;

/// <summary>A clock that moves only when told to, for the response store's expiry.</summary>
internal sealed class ManualTime : TimeProvider
{
    private readonly Lock _gate = new();
    private DateTimeOffset _now = new(2026, 11, 1, 0, 0, 0, TimeSpan.Zero);

    public override DateTimeOffset GetUtcNow()
    {
        lock (_gate)
        {
            return _now;
        }
    }

    public void Advance(TimeSpan by)
    {
        lock (_gate)
        {
            _now += by;
        }
    }
}
