# Upstream runs `git rev-parse`/`git rev-list` here to derive BUILD_COMMIT/
# BUILD_NUMBER. This tree is a vendored copy (no `.git` of its own) living
# inside the Minutist app repo, so that git detection would walk up and
# report the *app's* HEAD/commit-count instead of llama.cpp's — pin the
# values explicitly to the actual llama.cpp commit this fork is based on
# (see architecture/cross-cutting.md — "llama.cpp build + version policy").
set(BUILD_NUMBER 10200)
set(BUILD_COMMIT "5f55650a78f9")
set(BUILD_COMPILER "unknown")
set(BUILD_TARGET "unknown")

set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")

if(CMAKE_VS_PLATFORM_NAME)
    set(BUILD_TARGET ${CMAKE_VS_PLATFORM_NAME})
else()
    set(BUILD_TARGET "${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
