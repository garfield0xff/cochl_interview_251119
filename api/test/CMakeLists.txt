cmake_minimum_required(VERSION 3.18)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Test executable
add_executable(runtime_test
    runtime_test.cpp
)

target_include_directories(runtime_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Define PROJECT_ROOT for finding models
target_compile_definitions(runtime_test PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}/.."
)

target_link_libraries(runtime_test
    PRIVATE
        cochl_api
        GTest::gtest_main
)

# Link TFLite libraries if enabled
if(USE_TFLITE)
    target_link_libraries(runtime_test PRIVATE ${TFLITE_LIBRARIES})
endif()

# Link LibTorch libraries if enabled
if(USE_LIBTORCH)
    target_link_libraries(runtime_test PRIVATE ${TORCH_LIBRARIES})
endif()

# Add test
include(GoogleTest)
gtest_discover_tests(runtime_test)

# Set output directory
set_target_properties(runtime_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
)
