namespace Starlings.Roost.Engine.Caching;

/// <summary>
/// A pool slot as one number: the planner pins and parks by it. A recurrent group has two id spaces (live slots and
/// snapshots), so the key says which; otherwise snapshot 3 and live slot 3 of one group would be the same key.
/// </summary>
internal static class BlockKey
{
    private const int SnapshotFlag = 1 << 30;

    public static long Live(int group, int id) => ((long)group << 32) | (uint)id;

    public static long Snapshot(int group, int id) => ((long)(group | SnapshotFlag) << 32) | (uint)id;

    public static int Group(long key) => (int)(key >> 32) & ~SnapshotFlag;

    public static bool IsSnapshot(long key) => ((int)(key >> 32) & SnapshotFlag) != 0;

    public static int Id(long key) => (int)(key & 0xFFFFFFFF);
}
