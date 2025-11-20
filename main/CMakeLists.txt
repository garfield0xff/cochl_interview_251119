cmake_minimum_required(VERSION 3.18)
project(cochl_main VERSION 1.0.0)

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

# Find SDK library
set(SDK_ROOT "${CMAKE_SOURCE_DIR}/../sdk")
set(SDK_INCLUDE_DIR "${SDK_ROOT}/include")
set(SDK_BUILD_DIR "${SDK_ROOT}/build")
set(SDK_LIBRARY "${SDK_BUILD_DIR}/lib/libcochl_sdk.a")

if(NOT EXISTS ${SDK_LIBRARY})
    message(FATAL_ERROR "SDK library not found at: ${SDK_LIBRARY}. Please build SDK first.")
endif()

message(STATUS "Found SDK: ${SDK_LIBRARY}")

# Find cochl_api shared library
set(COCHL_API_LIB "${SDK_ROOT}/third_party/cochl_api/lib/libcochl_api.so")
if(NOT EXISTS ${COCHL_API_LIB})
    message(WARNING "cochl_api library not found at: ${COCHL_API_LIB}")
else()
    message(STATUS "Found cochl_api: ${COCHL_API_LIB}")
endif()

# Main executable
add_executable(main
    main.cpp
)

target_include_directories(main PRIVATE
    ${SDK_INCLUDE_DIR}
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

target_link_libraries(main PRIVATE
    ${SDK_LIBRARY}
    ${CMAKE_DL_LIBS}
    glog::glog
)

# Define library path as compile definition
target_compile_definitions(main PRIVATE
    COCHL_API_LIB_PATH="${COCHL_API_LIB}"
)

# Set output directory
set_target_properties(main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Configuration summary
message(STATUS "")
message(STATUS "=== Main Configuration ===")
message(STATUS "Architecture: ${ARCH_SUFFIX}")
message(STATUS "SDK library: ${SDK_LIBRARY}")
message(STATUS "==========================")
message(STATUS "")
