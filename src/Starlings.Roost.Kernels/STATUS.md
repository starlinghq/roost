# R-072 status: partial (CPU groundwork only)

R-072 (build plan section 10, streams C and F, M3): "Selector over conformance-passed variants per shape bucket and SM with a
persisted autotune cache keyed by SM, driver, shape bucket and variant digest; CUDA graphs recaptured when a selection changes; a
user library is qualified by running conformance against `CpuOps` and produces a digest-pinned `local-evidence` profile;
conformance-gated NVRTC compilation for missing specialisations with the cubin digest recorded; refusal (`ProfileUnsupported`)
stays the default when no qualified variant exists (bet X11)." This change is the CPU groundwork in `Starlings.Roost.Kernels`. No
contract (C1 to C9) changed. R-072 stays open.

## Delivered

- `ShapeBucket` (`ShapeBucket.cs`): op plus a canonical key over the `KernelRequirement` fields (dtypes, head_dim, layout,
  page_size, group_size, required flags) with batch and tokens rounded up to the next power of two, the bucketing `CudaOps` uses
  for GEMM plans. The SM is keyed separately.
- `VariantCandidate` (`VariantCandidate.cs`): a registered variant with a digest, SHA-256 over `roost-variant/1`, the numeric op
  id, the variant id and the SHA-256 the manifest records for the library holding it at the library's SM. A rebuilt library
  changes every digest, so qualification and measurements do not carry over.
- `VariantQualification` (`VariantQualification.cs`): the set of conformance passes, each naming op, variant id, exact digest,
  SM, one bucket or every bucket, and its evidence. An empty set qualifies nothing.
- `VariantSelector` (`VariantSelector.cs`): among the variants `KernelRegistry.Find` admits, only qualified ones are considered.
  With an autotune measurement for every qualified variant the lowest median wins; otherwise the default order is used and the
  unmeasured variants are listed. Ties and the default order are broken by variant id, then digest (ordinal), so the result does
  not depend on library load order. No qualified variant throws `ProfileUnsupportedException` naming each rejected variant and
  why. `VariantSelection.ChangedFrom` says whether a bucket's selection now runs other code (the recapture signal R-033 needs).
- `AutotuneCache` (`AutotuneCache.cs`): median time per (SM, driver version, op and shape bucket, variant id) with the variant
  digest stored and checked. Versioned JSON (`"format": "roost-autotune"`, `"version": 1`, source-generated
  `System.Text.Json`, entries written in key order so equal caches give identical files). A file for another SM or driver is
  dropped whole (`DeviceChanged`); a digest mismatch drops the entry on lookup; another version (`VersionMismatch`), unparseable
  or oversized content (`Corrupt`) opens empty and the next save replaces it; invalid or duplicate entries are dropped and
  counted. Saves write a temporary file in the same directory, flush it to disk and rename it over the old file.
- Tests: `VariantSelectorTests`, `VariantSelectorRegistryTests` and `AutotuneCacheTests` in
  `tests/Starlings.Roost.Kernels.Tests` (CPU only).

## Capability status

| Capability | Status | Evidence |
| --- | --- | --- |
| Variant selection over qualified variants, with refusal when none qualifies | experimental | CPU unit tests only; not wired into the worker (`CudaOpsSession` still takes the registry's first admitted variant) |
| Autotune cache (persistence, invalidation) | experimental | CPU unit tests only; no measurement has been taken on a GPU |
| Autotuned selection on any SM | untested | no autotune measurements exist; no GPU run |

## Still open (do not mark R-072 done)

- Conformance-gated qualification: running conformance against `CpuOps` to produce `QualifiedVariant` records, for the base
  library and for a user library, and the digest-pinned `local-evidence` profile.
- Autotune measurement on the GPU (timing each qualified variant per bucket through `tools/gpu-lease`) and wiring the selector
  and the cache into the worker; the X11 target (under 3% step time over a fixed choice) has no evidence.
- Recapturing CUDA graphs when a selection changes (needs R-033's capture; `VariantSelection.ChangedFrom` is the signal only).
- Conformance-gated NVRTC compilation of missing specialisations with the cubin digest recorded (`VariantCandidate.DigestOf`
  takes any code digest, but nothing compiles or records one yet).
- `GraphTarget.AvailableVariants` (C7) holds variant ids only, without digests or buckets; a C7 variant-selection pass would
  either look up digests through the registry or need a contract change through stream A. None was made here.
