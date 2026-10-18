// Every test here depends on a worker's threads (heartbeat, a step that sleeps, a spinning poll loop) running within tens
// of milliseconds of when they should. The worker runtime spins while idle and the tests spin while they wait, so on a
// two-core hosted runner test classes run in parallel starve each other's heartbeat and sleeping steps: a live worker was
// classified Stalled at the shipped 50 ms liveness window, and a 180 ms step overran a 250 ms deadline. The classes run
// one at a time; the stress test's seconds are the cost.
[assembly: Xunit.CollectionBehavior(Xunit.CollectionBehavior.CollectionPerAssembly)]
