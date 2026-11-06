# R-014 GPU comparison, rerun for #155 (2026-11-05)

The GPU half of R-014 on the ABI 1.3 libraries, against the same fixtures (`/home/mark/Projects/roost-evidence/R-014/ref`). Layout as in
the parent directory: `sm86/`, `sm89/`, `layers-sm<N>.json` and `.out`, `summary.json` and `summary.txt`. These are measurements, not an
adoption: `evidence/proposals/` is unchanged.

- **Kernel libraries.** `9e26f00c...` (SM 86) and `0edb2b2e...` (SM 89), CUDA 13.4.92, cuBLASLt 13.8.0.4, driver 615.78.08. The harness read
  the manifest with sha256 `ccb3c9c4...`, which differs from the final one only in the cuBLASLt pins and SM 89's status.
- **Result.** All 20 captures are byte-identical to the October ones in `../captures.sha256`, which came from the R-034 libraries on CUDA
  13.3.73, cuBLASLt 13.6.0.2 and driver 610.57. So every measured value equals the `[measured]` table of the proposal.
- **The one flip** is the one the proposal already lists: SM 89, `code`, forward 23, GPU 13 against reference 5199, reference margin 0.0096.
- **Commands.** `capture.sh` and `evaluate.sh` of `../scripts`, with paths changed and lease stream F, and `analyze.py` unchanged.
