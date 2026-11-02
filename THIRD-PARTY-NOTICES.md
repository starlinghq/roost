# Third-party notices

Roost's own code is MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). The kernel libraries and the native/ Rust libraries that Roost builds contain
code from the upstream projects below, fetched at build time by full commit SHA from the repositories named here (no
upstream source is committed; `PROVENANCE.md` has the revisions, paths taken and local changes, and
`kernels/manifest.toml` the pins and the licence per path). Binaries are release assets that carry these notices. Licence
texts below are copied from the pinned revisions.

Starling-family files reused here are listed in `PROVENANCE.md`.

## FlashInfer

- Repository: https://github.com/flashinfer-ai/flashinfer at `d84335d8c54e0c8f9f7e22faa15c7e34a58abf94`
- Taken: headers under `include/` as used by R-017 (`norm.cuh`: RMSNorm and QKRMSNorm; `activation.cuh`: the act-and-mul kernel; paged batch decode and batch prefill attention with their scheduler, `page.cuh` and the headers they include), unmodified. Not used: its RoPE kernel (R-017 measured it and wrote its own, kernels/README.md).
- Licence: Apache-2.0 (text below, under "Apache License 2.0"). The repository also carries `LICENSE.kda-for-kda.txt`
  (MIT, copyright 2026 KDA Team) for code adopted from that project; its text is reproduced under "KDA Team (MIT)".

## NVIDIA CUTLASS

- Repository: https://github.com/NVIDIA/cutlass at `098de2a652cf8f00fd70b2df54051c7eccbb855a`
- Taken: headers under `include/`, unmodified.
- Licence: BSD-3-Clause.

```text
Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Certain files within this repository are subject to separate licensing terms:

- The files located in the `python/CuTeDSL` directory are licensed under the
  NVIDIA End User License Agreement (EULA). Please refer to
  https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html
  for the full terms.
```

## vLLM

- Repository: https://github.com/vllm-project/vllm at `7879f3687b5b6a4be4754242e70b9c21cf7b4183`
- Taken: CUDA sources under `csrc/` (norm, rotary, MoE routing, Marlin), with the torch-free patch sets of
  `kernels/upstream/vllm.patches/` once an op adopts them. None is adopted yet.
- Licence: Apache-2.0 (text below).

## flash-linear-attention

- Repository: https://github.com/fla-org/flash-linear-attention at `329b534245d3d7bb27cc03a99e87bdc51fb28d06`
- Taken: the Triton kernels under `fla/` (Gated DeltaNet), compiled ahead of time to cubins at build time; no Python
  ships. Not adopted by an op yet (R-034).
- Licence: MIT.

```text
MIT License

Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## llguidance (native/roost-grammar)

- Repository: https://github.com/guidance-ai/llguidance at `6cef6ed9c973c2a155a85f2bb1996320af28b289` (crate 1.9.1)
- Taken: the crates `llguidance`, `toktrie` and `toktrie_hf_tokenizers` at that revision, as Cargo git dependencies of
  `native/roost-grammar` (pinned by `rev` in `Cargo.toml` and `Cargo.lock`), unmodified. No source is committed; the built
  `roost_grammar` library is a release asset (decision 22).
- Licence: MIT.

```text
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

### Crates from crates.io in the roost-grammar build

Every crates.io crate in the dependency graph, build-time proc-macro crates included, pinned by version and checksum in `native/roost-grammar/Cargo.lock` (the build uses `--locked`); listed from `cargo metadata`.
Where a crate offers a choice, Roost takes the licence in the last column. Apache-2.0 is the text under "Apache License 2.0"
below. Each crate's own copyright notice and licence file is in its source at the locked version; gathering them into the
release asset is part of the release workflow for native libraries, which is not built yet.

| Crate | Version | Licence (as declared) | Licence taken |
| --- | --- | --- | --- |
| `ahash` | 0.8.12 | MIT OR Apache-2.0 | MIT |
| `aho-corasick` | 1.1.5 | Unlicense OR MIT | MIT |
| `anyhow` | 1.0.104 | MIT OR Apache-2.0 | MIT |
| `base64` | 0.13.1 | MIT/Apache-2.0 | MIT |
| `bit-set` | 0.8.0 | Apache-2.0 OR MIT | MIT |
| `bit-vec` | 0.8.0 | Apache-2.0 OR MIT | MIT |
| `bitflags` | 2.13.2 | MIT OR Apache-2.0 | MIT |
| `borrow-or-share` | 0.2.4 | MIT-0 | MIT-0 |
| `bumpalo` | 3.20.3 | MIT OR Apache-2.0 | MIT |
| `bytemuck` | 1.25.2 | Zlib OR Apache-2.0 OR MIT | MIT |
| `bytemuck_derive` | 1.12.1 | Zlib OR Apache-2.0 OR MIT | MIT |
| `castaway` | 0.2.4 | MIT | MIT |
| `cfg-if` | 1.0.5 | MIT OR Apache-2.0 | MIT |
| `compact_str` | 0.9.1 | MIT | MIT |
| `crossbeam-deque` | 0.8.8 | MIT OR Apache-2.0 | MIT |
| `crossbeam-epoch` | 0.9.21 | MIT OR Apache-2.0 | MIT |
| `crossbeam-utils` | 0.8.23 | MIT OR Apache-2.0 | MIT |
| `darling` | 0.20.11 | MIT | MIT |
| `darling_core` | 0.20.11 | MIT | MIT |
| `darling_macro` | 0.20.11 | MIT | MIT |
| `dary_heap` | 0.3.9 | MIT OR Apache-2.0 | MIT |
| `derive_builder` | 0.20.2 | MIT OR Apache-2.0 | MIT |
| `derive_builder_core` | 0.20.2 | MIT OR Apache-2.0 | MIT |
| `derive_builder_macro` | 0.20.2 | MIT OR Apache-2.0 | MIT |
| `derivre` | 0.3.13 | MIT | MIT |
| `either` | 1.19.0 | MIT OR Apache-2.0 | MIT |
| `equivalent` | 1.0.3 | Apache-2.0 OR MIT | MIT |
| `esaxx-rs` | 0.1.10 | Apache-2.0 | Apache-2.0 |
| `fancy-regex` | 0.14.0 | MIT | MIT |
| `fluent-uri` | 0.3.2 | MIT | MIT |
| `fnv` | 1.0.7 | Apache-2.0 / MIT | MIT |
| `foldhash` | 0.2.0 | Zlib | Zlib |
| `getrandom` | 0.3.4 | MIT OR Apache-2.0 | MIT |
| `hashbrown` | 0.17.1 | MIT OR Apache-2.0 | MIT |
| `heck` | 0.5.0 | MIT OR Apache-2.0 | MIT |
| `ident_case` | 1.0.1 | MIT/Apache-2.0 | MIT |
| `indexmap` | 2.14.2 | Apache-2.0 OR MIT | MIT |
| `itertools` | 0.14.0 | MIT OR Apache-2.0 | MIT |
| `itoa` | 1.0.18 | MIT OR Apache-2.0 | MIT |
| `js-sys` | 0.3.106 | MIT OR Apache-2.0 | MIT |
| `libc` | 0.2.191 | MIT OR Apache-2.0 | MIT |
| `lock_api` | 0.4.14 | MIT OR Apache-2.0 | MIT |
| `log` | 0.4.34 | MIT OR Apache-2.0 | MIT |
| `macro_rules_attribute` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | MIT |
| `macro_rules_attribute-proc_macro` | 0.2.3 | Apache-2.0 OR MIT OR Zlib | MIT |
| `memchr` | 2.8.3 | Unlicense OR MIT | MIT |
| `minimal-lexical` | 0.2.1 | MIT/Apache-2.0 | MIT |
| `monostate` | 0.1.18 | MIT OR Apache-2.0 | MIT |
| `monostate-impl` | 0.1.18 | MIT OR Apache-2.0 | MIT |
| `nom` | 7.1.3 | MIT | MIT |
| `once_cell` | 1.21.4 | MIT OR Apache-2.0 | MIT |
| `parking_lot` | 0.12.5 | MIT OR Apache-2.0 | MIT |
| `parking_lot_core` | 0.9.12 | MIT OR Apache-2.0 | MIT |
| `paste` | 1.0.15 | MIT OR Apache-2.0 | MIT |
| `pastey` | 0.2.3 | MIT OR Apache-2.0 | MIT |
| `percent-encoding` | 2.3.2 | MIT OR Apache-2.0 | MIT |
| `ppv-lite86` | 0.2.21 | MIT OR Apache-2.0 | MIT |
| `proc-macro2` | 1.0.107 | MIT OR Apache-2.0 | MIT |
| `quote` | 1.0.47 | MIT OR Apache-2.0 | MIT |
| `r-efi` | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later | MIT |
| `rand` | 0.9.5 | MIT OR Apache-2.0 | MIT |
| `rand_chacha` | 0.9.0 | MIT OR Apache-2.0 | MIT |
| `rand_core` | 0.9.5 | MIT OR Apache-2.0 | MIT |
| `rayon` | 1.12.0 | MIT OR Apache-2.0 | MIT |
| `rayon-cond` | 0.4.0 | Apache-2.0/MIT | MIT |
| `rayon-core` | 1.13.0 | MIT OR Apache-2.0 | MIT |
| `redox_syscall` | 0.5.18 | MIT | MIT |
| `ref-cast` | 1.0.27 | MIT OR Apache-2.0 | MIT |
| `ref-cast-impl` | 1.0.27 | MIT OR Apache-2.0 | MIT |
| `referencing` | 0.29.1 | MIT | MIT |
| `regex` | 1.13.1 | MIT OR Apache-2.0 | MIT |
| `regex-automata` | 0.4.18 | MIT OR Apache-2.0 | MIT |
| `regex-syntax` | 0.8.11 | MIT OR Apache-2.0 | MIT |
| `rustversion` | 1.0.23 | MIT OR Apache-2.0 | MIT |
| `ryu` | 1.0.23 | Apache-2.0 OR BSL-1.0 | Apache-2.0 |
| `scopeguard` | 1.2.0 | MIT OR Apache-2.0 | MIT |
| `serde` | 1.0.229 | MIT OR Apache-2.0 | MIT |
| `serde_core` | 1.0.229 | MIT OR Apache-2.0 | MIT |
| `serde_derive` | 1.0.229 | MIT OR Apache-2.0 | MIT |
| `serde_json` | 1.0.154 | MIT OR Apache-2.0 | MIT |
| `smallvec` | 1.16.3 | MIT OR Apache-2.0 | MIT |
| `spm_precompiled` | 0.1.4 | Apache-2.0 | Apache-2.0 |
| `static_assertions` | 1.1.0 | MIT OR Apache-2.0 | MIT |
| `strsim` | 0.11.1 | MIT | MIT |
| `strum` | 0.28.0 | MIT | MIT |
| `strum_macros` | 0.28.0 | MIT | MIT |
| `syn` | 2.0.119 | MIT OR Apache-2.0 | MIT |
| `syn` | 3.0.9 | MIT OR Apache-2.0 | MIT |
| `thiserror` | 2.0.21 | MIT OR Apache-2.0 | MIT |
| `thiserror-impl` | 2.0.21 | MIT OR Apache-2.0 | MIT |
| `tokenizers` | 0.21.4 | Apache-2.0 | Apache-2.0 |
| `unicode-ident` | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 | MIT AND Unicode-3.0 |
| `unicode-normalization-alignments` | 0.1.12 | MIT/Apache-2.0 | MIT |
| `unicode-segmentation` | 1.13.3 | MIT OR Apache-2.0 | MIT |
| `unicode_categories` | 0.1.1 | MIT OR Apache-2.0 | MIT |
| `version_check` | 0.9.5 | MIT/Apache-2.0 | MIT |
| `wasip2` | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | MIT |
| `wasm-bindgen` | 0.2.129 | MIT OR Apache-2.0 | MIT |
| `wasm-bindgen-macro` | 0.2.129 | MIT OR Apache-2.0 | MIT |
| `wasm-bindgen-macro-support` | 0.2.129 | MIT OR Apache-2.0 | MIT |
| `wasm-bindgen-shared` | 0.2.129 | MIT OR Apache-2.0 | MIT |
| `windows-link` | 0.2.1 | MIT OR Apache-2.0 | MIT |
| `wit-bindgen` | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | MIT |
| `zerocopy` | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | MIT |
| `zerocopy-derive` | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT | MIT |
| `zmij` | 1.0.23 | MIT | MIT |

## KDA Team (MIT)

Licence file shipped with FlashInfer (`LICENSE.kda-for-kda.txt`).

```text
MIT License

Copyright (c) 2026 KDA Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Components bundled in FlashInfer

FlashInfer's `LICENSE` lists code it adapted from other projects: BSD-3-Clause for the Hopper attention headers
`include/flashinfer/attention/hopper/{epilogue,mainloop,kernel_traits,named_barrier,tile_scheduler,utils,block_sparse_gather}.cuh`
(adapted from FlashAttention 3 and CUTLASS), and MIT for `3rdparty/spdlog` and its bundled fmt, which Roost does not
build. The FlashAttention 3 text, from `licenses/LICENSE.flashattention3.txt` at the pinned revision:

```text
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Apache License 2.0

The text of the licence for FlashInfer and vLLM, copied from vLLM's `LICENSE` at the revision above. FlashInfer's
`LICENSE` is the same text followed by the list of bundled components summarised in the previous section.

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## NVIDIA CUDA toolkit

The kernel libraries link the shared CUDA runtime, and Gemm and LmHead call cuBLASLt (S2, R-017). Both come from the CUDA toolkit
under the NVIDIA CUDA EULA, linked dynamically and not part of the kernel assets; their redistribution terms belong to the decision 5 licence review and are not asserted here.
