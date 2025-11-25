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
add_executable(api_test
    api_test.cpp
)

target_include_directories(api_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Define PROJECT_ROOT for finding models
target_compile_definitions(api_test PRIVATE
    PROJECT_ROOT="${CMAKE_SOURCE_DIR}/.."
)

target_link_libraries(api_test
    PRIVATE
        cochl_api
        GTest::gtest_main
)

# Link TFLite libraries if enabled
if(USE_TFLITE)
    target_link_libraries(api_test PRIVATE ${TFLITE_LIBRARIES})
endif()

# Link LibTorch libraries if enabled
if(USE_LIBTORCH)
    target_link_libraries(api_test PRIVATE ${TORCH_LIBRARIES})
endif()

# Link TVM libraries if enabled
if(USE_TVM)
    target_link_libraries(api_test PRIVATE ${TVM_LIBRARY})
endif()

# Add test
include(GoogleTest)
gtest_discover_tests(api_test)

# Set output directory
set_target_properties(api_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test
)
