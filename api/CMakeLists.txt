cmake_minimum_required(VERSION 3.18)
project(cochl_api VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(ARCH "arm64")
    message(STATUS "Detected ARM64 architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH "amd64")
    message(STATUS "Detected AMD64 architecture")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Set third_party path based on architecture
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/../third_party")
set(LIBTORCH_DIR "${THIRD_PARTY_DIR}/libtorch-${ARCH}")
set(TFLITE_DIR "${THIRD_PARTY_DIR}/tflite-${ARCH}")
set(TVM_DIR "${THIRD_PARTY_DIR}/tvm-${ARCH}")

# Options
option(USE_LIBTORCH "Build with LibTorch support" ON)
option(USE_TFLITE "Build with TensorFlow Lite support" ON)
option(USE_TVM "Build with TVM support" OFF)
option(USE_CUSTOM "Build with Custom runtime support" ON)
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# pybind11 (header-only)
if(EXISTS "${THIRD_PARTY_DIR}/pybind11/pybind11.h")
    include_directories(${THIRD_PARTY_DIR}/pybind11)
    message(STATUS "Found pybind11: ${THIRD_PARTY_DIR}/pybind11")
endif()

# Find LibTorch
if(USE_LIBTORCH)
    if(EXISTS "${LIBTORCH_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
        find_package(Torch REQUIRED)
        message(STATUS "Found LibTorch: ${LIBTORCH_DIR}")
        message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
        add_compile_definitions(USE_LIBTORCH)
    else()
        message(WARNING "LibTorch not found at ${LIBTORCH_DIR}, disabling LibTorch support")
        set(USE_LIBTORCH OFF)
    endif()
endif()

# Find TensorFlow Lite
if(USE_TFLITE)
    if(EXISTS "${TFLITE_DIR}")
        set(TFLITE_INCLUDE_DIR "${TFLITE_DIR}/include")
        set(TFLITE_LIB_DIR "${TFLITE_DIR}/lib")

        # Main TensorFlow Lite library
        find_library(TFLITE_MAIN_LIB tensorflow-lite PATHS ${TFLITE_LIB_DIR} NO_DEFAULT_PATH)

        if(TFLITE_MAIN_LIB)
            # Build TFLITE_LIBRARIES with proper link order
            # Use --start-group/--end-group to handle circular dependencies
            file(GLOB TFLITE_ADDITIONAL_LIBS "${TFLITE_LIB_DIR}/*.a")
            list(REMOVE_ITEM TFLITE_ADDITIONAL_LIBS ${TFLITE_MAIN_LIB})

            set(TFLITE_LIBRARIES
                -Wl,--start-group
                ${TFLITE_MAIN_LIB}
                ${TFLITE_ADDITIONAL_LIBS}
                -Wl,--end-group
            )

            message(STATUS "Found TensorFlow Lite: ${TFLITE_DIR}")
            message(STATUS "TFLite main library: ${TFLITE_MAIN_LIB}")
            add_compile_definitions(USE_TFLITE)
        else()
            message(WARNING "TFLite main library not found, disabling TFLite support")
            set(USE_TFLITE OFF)
        endif()
    else()
        message(WARNING "TFLite not found at ${TFLITE_DIR}, disabling TFLite support")
        set(USE_TFLITE OFF)
    endif()
endif()

# Find TVM
if(USE_TVM)
    if(EXISTS "${TVM_DIR}")
        set(TVM_INCLUDE_DIR "${TVM_DIR}/include")
        set(TVM_LIB_DIR "${TVM_DIR}/lib")
        find_library(TVM_LIBRARY NAMES tvm PATHS ${TVM_LIB_DIR} NO_DEFAULT_PATH)

        if(TVM_LIBRARY)
            message(STATUS "Found TVM: ${TVM_DIR}")
            add_compile_definitions(USE_TVM)
        else()
            message(WARNING "TVM library not found, disabling TVM support")
            set(USE_TVM OFF)
        endif()
    else()
        message(WARNING "TVM not found at ${TVM_DIR}, disabling TVM support")
        set(USE_TVM OFF)
    endif()
endif()

# Source files
set(RUNTIME_SOURCES
    src/runtime/runtime_manager.cpp
)

if(USE_TFLITE)
    list(APPEND RUNTIME_SOURCES src/runtime/tf_runtime.cpp)
endif()

if(USE_LIBTORCH)
    list(APPEND RUNTIME_SOURCES src/runtime/torch_runtime.cpp)
endif()

if(USE_CUSTOM)
    list(APPEND RUNTIME_SOURCES src/runtime/custom_runtime.cpp)
    add_compile_definitions(USE_CUSTOM)
endif()

set(API_SOURCES
    src/cochl_api.cpp
    src/cochl_api_c.cpp
    src/error/api_error.cpp
    ${RUNTIME_SOURCES}
)

# Create library
if(BUILD_SHARED_LIBS)
    add_library(cochl_api SHARED ${API_SOURCES})
else()
    add_library(cochl_api STATIC ${API_SOURCES})
endif()

# Include directories
target_include_directories(cochl_api
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link TFLite
if(USE_TFLITE)
    target_include_directories(cochl_api PRIVATE ${TFLITE_INCLUDE_DIR})
    if(BUILD_SHARED_LIBS)
        target_link_libraries(cochl_api PRIVATE ${TFLITE_LIBRARIES})
    else()
        # For static library, embed TFLite symbols using whole-archive
        target_link_libraries(cochl_api PRIVATE
            -Wl,--whole-archive
            ${TFLITE_LIBRARIES}
            -Wl,--no-whole-archive
        )
    endif()
endif()

# Link LibTorch
if(USE_LIBTORCH)
    if(BUILD_SHARED_LIBS)
        target_link_libraries(cochl_api PRIVATE ${TORCH_LIBRARIES})
    else()
        # For static library, link LibTorch (it's already dynamic, so just link normally)
        target_link_libraries(cochl_api PRIVATE ${TORCH_LIBRARIES})
    endif()
    # Copy LibTorch dlls on Windows
    if(MSVC)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        add_custom_command(TARGET cochl_api POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:cochl_api>)
    endif()
endif()

# Link TVM
if(USE_TVM)
    target_include_directories(cochl_api PRIVATE ${TVM_INCLUDE_DIR})
    target_link_libraries(cochl_api PRIVATE ${TVM_LIBRARY})
endif()

# Set output directory
set_target_properties(cochl_api PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install rules
install(TARGETS cochl_api
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)

# Build tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Cochl API Configuration ===")
message(STATUS "Architecture: ${ARCH}")
message(STATUS "LibTorch support: ${USE_LIBTORCH}")
message(STATUS "TFLite support: ${USE_TFLITE}")
message(STATUS "TVM support: ${USE_TVM}")
message(STATUS "Custom runtime support: ${USE_CUSTOM}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "================================")
message(STATUS "")
