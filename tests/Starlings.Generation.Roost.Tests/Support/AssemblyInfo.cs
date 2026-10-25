// The worker runtime spins while idle and each client's driver thread spins while a step runs; on a two-core hosted runner,
// test classes in parallel starve each other's heartbeats (as in Starlings.Roost.WorkerHost.Tests). The classes run one at a time.
[assembly: Xunit.CollectionBehavior(Xunit.CollectionBehavior.CollectionPerAssembly)]
