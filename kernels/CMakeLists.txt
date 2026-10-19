# Roost kernel library (R-016): one shared library per SM, each exporting the C1 registration table.
#
#   python3 -m venv ... ; cmake -S kernels -B kernels/build -G Ninja [-DROOST_SM_LIST=86]
#   cmake --build kernels/build
#
# The SM list defaults to the manifest's (kernels/manifest.toml); a developer builds one SM with -DROOST_SM_LIST=86,
# and CI builds them all. Upstream sources are fetched by full SHA at configure time (cmake/RoostUpstreams.cmake) into
# ROOST_UPSTREAM_DIR; none is committed. Binaries are release assets: this build writes only under its build directory.

cmake_minimum_required(VERSION 3.24)
project(roost_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared) # S1: one shared CUDA runtime for the host and every kernel library
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

get_filename_component(ROOST_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(ROOST_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/manifest.toml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${ROOST_MANIFEST}")

# --- Manifest ------------------------------------------------------------------------------------------------------
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter) # tomllib
set(ROOST_MANIFEST_CMAKE "${CMAKE_BINARY_DIR}/roost_manifest.cmake")
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -I "${CMAKE_CURRENT_LIST_DIR}/tools/manifest_cmake.py" "${ROOST_MANIFEST}"
  OUTPUT_FILE "${ROOST_MANIFEST_CMAKE}" RESULT_VARIABLE manifest_rc)
if(NOT manifest_rc EQUAL 0)
  message(FATAL_ERROR "cannot read ${ROOST_MANIFEST}")
endif()
include("${ROOST_MANIFEST_CMAKE}")

set(ROOST_SM_LIST "${ROOST_MANIFEST_SM_LIST}" CACHE STRING "SMs to build, as major*10+minor (default: the manifest's list)")
foreach(arch IN LISTS ROOST_SM_LIST)
  if(NOT arch IN_LIST ROOST_MANIFEST_SM_LIST)
    message(FATAL_ERROR "SM ${arch} is not in the manifest's SM list (${ROOST_MANIFEST_SM_LIST})")
  endif()
endforeach()

# --- Upstream sources ----------------------------------------------------------------------------------------------
option(ROOST_FETCH_UPSTREAMS "Fetch the manifest's upstream sources by SHA at configure time" ON)
set(ROOST_UPSTREAM_DIR "${CMAKE_BINARY_DIR}/upstream" CACHE PATH "Where upstream checkouts live")
if(ROOST_FETCH_UPSTREAMS)
  include("${CMAKE_CURRENT_LIST_DIR}/cmake/RoostUpstreams.cmake")
  roost_fetch_all_upstreams()
endif()

# The M0 ops compile against FlashInfer's headers and link cuBLASLt (R-017), so the fetch cannot be skipped.
find_package(CUDAToolkit REQUIRED)
if(NOT DEFINED ROOST_FLASHINFER_DIR OR NOT EXISTS "${ROOST_FLASHINFER_DIR}/include/flashinfer")
  message(FATAL_ERROR "the M0 ops need the pinned FlashInfer headers: configure with -DROOST_FETCH_UPSTREAMS=ON")
endif()

# --- Libraries, one per SM -----------------------------------------------------------------------------------------
set(ROOST_LIBRARY_SOURCES src/registry.cu src/residual.cu src/m0_registry.cu src/m0_ops.cu src/paged_attention.cu src/gemm.cu)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/RoostHybrid.cmake") # R-034: appends src/ops/*.cu, defines roost_hybrid_attach()
set(ROOST_TARGETS "")
foreach(arch IN LISTS ROOST_SM_LIST)
  set(target roost_kernels_sm${arch})
  add_library(${target} SHARED ${ROOST_LIBRARY_SOURCES})
  target_include_directories(${target} PRIVATE include src)
  # R-017: the M0 ops use FlashInfer's headers (SYSTEM: -Wall -Werror applies to our code, not the upstream's) and cuBLASLt.
  target_include_directories(${target} SYSTEM PRIVATE "${ROOST_FLASHINFER_DIR}/include")
  target_link_libraries(${target} PRIVATE CUDA::cublasLt)
  target_compile_definitions(${target} PRIVATE ROOST_SM=${arch} ROOST_BUILDING_LIBRARY=1)
  # SASS for exactly this SM and no PTX: a library for another SM is a different file, so a missing SM is a typed
  # startup failure (R-023) and the digest in the manifest is per SM.
  set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "${arch}-real" PREFIX "lib" OUTPUT_NAME ${target})
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wall,-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${ROOST_REPO_ROOT}=.>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffile-prefix-map=${ROOST_UPSTREAM_DIR}=upstream>)
  if(UNIX)
    # Same inputs, same bytes: no line info (it records source mtimes), no build id and no local symbols (nvcc names
    # them after the process id), so the digest recorded in the manifest can be reproduced.
    target_link_options(${target} PRIVATE -Wl,--build-id=none -Wl,--strip-all)
  endif()
  roost_hybrid_attach(${target} ${arch})
  list(APPEND ROOST_TARGETS ${target})
endforeach()

# --- Registration and stub check (needs a GPU; run through tools/gpu-lease, see kernels/README.md) ------------------
add_executable(roost_stub_check tests/stub_check.cpp)
target_include_directories(roost_stub_check PRIVATE include)
target_link_libraries(roost_stub_check PRIVATE CUDA::cudart ${CMAKE_DL_LIBS})
set_target_properties(roost_stub_check PROPERTIES BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR}")
add_dependencies(roost_stub_check ${ROOST_TARGETS})

message(STATUS "roost: SM list ${ROOST_SM_LIST}; CUDA ${CMAKE_CUDA_COMPILER_VERSION}")
