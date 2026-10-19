# Hybrid ops (R-034): CausalConv1d, the recurrent state ops and, when a Triton AOT interpreter is given, GatedDeltaRule.
# Appends the op sources to ROOST_LIBRARY_SOURCES and defines roost_hybrid_attach(<target> <sm>), called once per SM
# target by kernels/CMakeLists.txt.

list(APPEND ROOST_LIBRARY_SOURCES
  src/ops/hybrid_entries.cu src/ops/causal_conv1d.cu src/ops/recurrent_state.cu)

function(roost_hybrid_attach target arch)
endfunction()
