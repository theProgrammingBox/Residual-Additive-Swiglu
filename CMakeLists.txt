cmake_minimum_required(VERSION 3.18)
project(CudaProject LANGUAGES CXX CUDA)

# --- Main Executable ---
add_executable(app
    main.cu
    src/GridworldEnv.cpp
)

# --- Testing ---
# Enable the testing infrastructure
enable_testing()

# Download and configure Google Test automatically
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# Create a new executable for our tests
add_executable(run_tests
    tests/test_env.cpp
    src/GridworldEnv.cpp
)

# Link our test executable against the Google Test library
target_link_libraries(run_tests GTest::gtest_main)

# Tell CMake's test runner ("ctest") about our test executable
include(GoogleTest)
gtest_discover_tests(run_tests)