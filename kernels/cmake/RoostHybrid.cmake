# Hybrid ops (R-034): CausalConv1d, the recurrent state ops and the Qwen3.5 gating ops always; GatedDeltaRule when a Triton AOT interpreter is given.
# Appends the op sources to ROOST_LIBRARY_SOURCES and defines roost_hybrid_attach(<target> <sm>), called once per SM
# target by kernels/CMakeLists.txt.
#
# GatedDeltaRule embeds Triton AOT cubins (docs/m0/roost-s3-triton-aot-gdn.md, decision 23). Triton and Python run at
# build time only, and the build host needs a visible CUDA device (kernels/aot/build_gdn.py says why), so the cubins are
# opt-in and the GPU-less CI build compiles the library without them (GatedDeltaRule is then not registered):
#
#   -DROOST_TRITON_PYTHON=<python of the [triton_aot] closure in manifest.toml>
#   -DROOST_FLA_DIR=<flash-linear-attention checkout at the manifest's SHA>   (default: the fetched upstream)
#
# Run that build through tools/gpu-lease: the AOT step is a GPU process.

list(APPEND ROOST_LIBRARY_SOURCES
  src/ops/hybrid_entries.cu src/ops/causal_conv1d.cu src/ops/recurrent_state.cu src/ops/gating.cu)

set(ROOST_TRITON_PYTHON "" CACHE FILEPATH "Python with Triton and torch for the GatedDeltaRule AOT kernels (empty: build without them)")
set(ROOST_FLA_DIR "" CACHE PATH "flash-linear-attention checkout at the manifest's SHA (default: the fetched upstream)")

if(ROOST_TRITON_PYTHON)
  if(NOT ROOST_FLA_DIR)
    if(DEFINED ROOST_FLASH_LINEAR_ATTENTION_DIR)
      set(ROOST_FLA_DIR "${ROOST_FLASH_LINEAR_ATTENTION_DIR}")
    else()
      message(FATAL_ERROR "ROOST_TRITON_PYTHON is set: give ROOST_FLA_DIR, or leave ROOST_FETCH_UPSTREAMS on")
    endif()
  endif()
  set(ROOST_AOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../aot")
  message(STATUS "roost: GatedDeltaRule AOT kernels from ${ROOST_TRITON_PYTHON}, FLA ${ROOST_FLA_DIR}")
endif()

function(roost_hybrid_attach target arch)
  if(NOT ROOST_TRITON_PYTHON)
    return()
  endif()
  set(generated "${CMAKE_BINARY_DIR}/aot/roost_gdn_cubins_sm${arch}.cpp")
  # A cold Triton cache per SM, inside the build tree: the cubins must not depend on a previous build's cache.
  set(cache "${CMAKE_BINARY_DIR}/aot/triton-cache-sm${arch}")
  add_custom_command(
    OUTPUT "${generated}"
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${cache}"
    COMMAND "${CMAKE_COMMAND}" -E env "TRITON_CACHE_DIR=${cache}"
            "${ROOST_TRITON_PYTHON}" -I "${ROOST_AOT_DIR}/build_gdn.py" "${ROOST_FLA_DIR}"
            "${ROOST_AOT_DIR}/gdn_specs.json" ${arch} "${generated}"
    DEPENDS "${ROOST_AOT_DIR}/build_gdn.py" "${ROOST_AOT_DIR}/gdn_specs.json"
    COMMENT "Triton AOT: GatedDeltaRule cubins for sm_${arch}"
    VERBATIM)
  target_sources(${target} PRIVATE src/ops/gated_delta_rule.cu "${generated}")
  target_include_directories(${target} PRIVATE src/ops)
  target_compile_definitions(${target} PRIVATE ROOST_HYBRID_GDN=1)
endfunction()
