# Fetch the manifest's upstream sources by full commit SHA and apply their patch sets (R-016).
#
# Inputs: ROOST_UPSTREAM_DIR (cache, default <build>/upstream) and the variables manifest_cmake.py generated.
# Output: ROOST_<NAME>_DIR for each upstream, NAME upper-cased with '-' as '_', for example ROOST_FLASHINFER_DIR.
#
# A checkout is keyed by name and SHA. It is fetched with `git fetch --depth 1 origin <sha>` and refused unless
# HEAD is exactly that SHA, so a moved tag or branch can never change a build. Patches are applied in file-name order
# onto a clean checkout and each must match the SHA-256 the manifest records.

find_package(Git REQUIRED)

function(roost_run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    string(REPLACE ";" " " cmd "${ARGN}")
    message(FATAL_ERROR "command failed (${rc}): ${cmd}\n${out}${err}")
  endif()
endfunction()

function(roost_fetch_upstream index)
  set(name "${ROOST_UPSTREAM_${index}_NAME}")
  set(repo "${ROOST_UPSTREAM_${index}_REPO}")
  set(sha "${ROOST_UPSTREAM_${index}_SHA}")
  string(LENGTH "${sha}" sha_length)
  if(NOT sha MATCHES "^[0-9a-f]+$" OR NOT sha_length EQUAL 40)
    message(FATAL_ERROR "upstream ${name}: '${sha}' is not a full 40-character commit SHA")
  endif()
  set(dir "${ROOST_UPSTREAM_DIR}/${name}-${sha}")
  if(NOT EXISTS "${dir}/.git")
    file(MAKE_DIRECTORY "${dir}")
    roost_run("${GIT_EXECUTABLE}" -C "${dir}" init -q)
    roost_run("${GIT_EXECUTABLE}" -C "${dir}" remote add origin "${repo}")
  endif()
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse --verify -q "${sha}^{commit}"
                  RESULT_VARIABLE have OUTPUT_QUIET ERROR_QUIET)
  if(NOT have EQUAL 0)
    message(STATUS "roost: fetching ${name} ${sha}")
    roost_run("${GIT_EXECUTABLE}" -C "${dir}" fetch -q --depth 1 origin "${sha}")
  endif()
  # Clean checkout of exactly the pinned commit, then the patch set.
  roost_run("${GIT_EXECUTABLE}" -C "${dir}" checkout -q --force "${sha}")
  roost_run("${GIT_EXECUTABLE}" -C "${dir}" clean -q -fdx)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse HEAD OUTPUT_VARIABLE head
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT head STREQUAL sha)
    message(FATAL_ERROR "upstream ${name}: HEAD is ${head}, expected ${sha}")
  endif()
  foreach(entry IN LISTS ROOST_UPSTREAM_${index}_PATCHES)
    string(REPLACE "|" ";" parts "${entry}")
    list(GET parts 0 patch_rel)
    list(GET parts 1 patch_sha256)
    set(patch "${ROOST_REPO_ROOT}/${patch_rel}")
    file(SHA256 "${patch}" actual)
    if(NOT actual STREQUAL patch_sha256)
      message(FATAL_ERROR "patch ${patch_rel}: SHA-256 ${actual} differs from the manifest's ${patch_sha256}")
    endif()
    roost_run("${GIT_EXECUTABLE}" -C "${dir}" apply --whitespace=nowarn "${patch}")
  endforeach()
  string(TOUPPER "${name}" upper)
  string(REPLACE "-" "_" upper "${upper}")
  set(ROOST_${upper}_DIR "${dir}" PARENT_SCOPE)
  message(STATUS "roost: ${name} ${sha} -> ${dir}")
endfunction()

function(roost_fetch_all_upstreams)
  list(LENGTH ROOST_UPSTREAMS count)
  math(EXPR last "${count} - 1")
  foreach(i RANGE ${last})
    roost_fetch_upstream(${i})
    string(TOUPPER "${ROOST_UPSTREAM_${i}_NAME}" upper)
    string(REPLACE "-" "_" upper "${upper}")
    set(ROOST_${upper}_DIR "${ROOST_${upper}_DIR}" PARENT_SCOPE)
  endforeach()
endfunction()
