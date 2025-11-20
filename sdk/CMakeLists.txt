cmake_minimum_required(VERSION 3.18)
project(cochl_sdk VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(ARCH_SUFFIX "arm64")
    message(STATUS "Detected ARM64 architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_SUFFIX "amd64")
    message(STATUS "Detected AMD64 architecture")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# cochl_api headers (for interface only, no linking)
set(COCHL_API_ROOT "${CMAKE_SOURCE_DIR}/third_party/cochl_api")
set(COCHL_API_INCLUDE_DIR "${COCHL_API_ROOT}/include")

if(NOT EXISTS ${COCHL_API_INCLUDE_DIR})
    message(FATAL_ERROR "cochl_api headers not found at: ${COCHL_API_INCLUDE_DIR}")
endif()

message(STATUS "Found cochl_api headers: ${COCHL_API_INCLUDE_DIR}")

# SDK library
add_library(cochl_sdk
    src/inference_engine.cpp
    src/error/sdk_error.cpp
    src/api/cochl_api.cpp
)

target_include_directories(cochl_sdk PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${COCHL_API_INCLUDE_DIR}
)

# glog: Use FetchContent to download and build
include(FetchContent)
FetchContent_Declare(
    glog
    GIT_REPOSITORY https://github.com/google/glog.git
    GIT_TAG v0.6.0
)
set(WITH_GFLAGS OFF CACHE BOOL "Use gflags")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
FetchContent_MakeAvailable(glog)

# Link dl library for dlopen/dlsym
target_link_libraries(cochl_sdk PUBLIC
    ${CMAKE_DL_LIBS}
    glog::glog
)

# Set output directory
set_target_properties(cochl_sdk PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Configuration summary
message(STATUS "")
message(STATUS "=== Cochl SDK Configuration ===")
message(STATUS "Architecture: ${ARCH_SUFFIX}")
message(STATUS "Dynamic loading: YES (dlopen)")
message(STATUS "Build type: ${BUILD_SHARED_LIBS}")
message(STATUS "================================")
message(STATUS "")
