cmake_minimum_required(VERSION 3.14)

project(GlobalOptimizers
    VERSION 0.1.0
    DESCRIPTION "Header-only library of global and local optimizers based on Eigen"
    LANGUAGES CXX)

# -- build type ---------------------------------------------------------------

# These optimizers are Eigen-heavy and unusably slow without optimization: an
# -O0 build of the EGO benchmark sweep is tens of times slower than -O3 (and
# Eigen's assertions are active). Default single-config generators to Release
# rather than to the empty (no optimization) build type.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()

# -- Eigen --------------------------------------------------------------------

# L-BFGS-B uses Eigen's indexed views, which need Eigen >= 3.4
find_package(Eigen3 3.4 QUIET)

if (NOT TARGET Eigen3::Eigen)
    # fall back to a sibling checkout of the Eigen repository
    set(GLOBOPT_EIGEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../eigen" CACHE PATH "Path to an Eigen source tree")

    if (EXISTS "${GLOBOPT_EIGEN_DIR}/Eigen/Dense")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLOBOPT_EIGEN_DIR}")
    else()
        message(FATAL_ERROR "Eigen not found. Install Eigen3 or set GLOBOPT_EIGEN_DIR to an Eigen source tree.")
    endif()
endif()

# -- library ------------------------------------------------------------------

add_library(globopt INTERFACE)
add_library(globopt::globopt ALIAS globopt)

target_include_directories(globopt INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(globopt INTERFACE cxx_std_17)
target_link_libraries(globopt INTERFACE Eigen3::Eigen)

# -- tests and examples -------------------------------------------------------

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    option(GLOBOPT_BUILD_TESTS "Build GlobalOptimizers tests" ON)
    option(GLOBOPT_BUILD_EXAMPLES "Build GlobalOptimizers examples" ON)
else()
    option(GLOBOPT_BUILD_TESTS "Build GlobalOptimizers tests" OFF)
    option(GLOBOPT_BUILD_EXAMPLES "Build GlobalOptimizers examples" OFF)
endif()

if (GLOBOPT_BUILD_TESTS)
    enable_testing()

    add_executable(test_optimizers tests/test_optimizers.cpp)
    target_link_libraries(test_optimizers PRIVATE globopt::globopt)
    if (MSVC)
        target_compile_options(test_optimizers PRIVATE /bigobj)
    endif()
    add_test(NAME optimizers COMMAND test_optimizers)
endif()

if (GLOBOPT_BUILD_EXAMPLES)
    add_executable(lbfgs_example examples/lbfgs_example.cpp)
    target_link_libraries(lbfgs_example PRIVATE globopt::globopt)
endif()

if (GLOBOPT_BUILD_TESTS)
    add_executable(run_ampgo_benchmarks benchmarks/run_ampgo.cpp)
    target_link_libraries(run_ampgo_benchmarks PRIVATE globopt::globopt)
    if (MSVC)
        target_compile_options(run_ampgo_benchmarks PRIVATE /bigobj)
    endif()

    # Smoke-test the benchmark driver on a small slice of the suite: enough to
    # catch a broken optimizer or driver, fast enough to run with the unit
    # tests. Full sweeps are run by hand (see the README).
    add_test(NAME benchmark_subset_ego
        COMMAND run_ampgo_benchmarks 12345 EGO --stride 40 --budget 60 --refit-interval 5)
    add_test(NAME benchmark_subset_lipo
        COMMAND run_ampgo_benchmarks 12345 LIPO --stride 40 --budget 200)
    add_test(NAME benchmark_scalable_cmaes
        COMMAND run_ampgo_benchmarks 12345 CMA-ES --scalable 5,10 --budget 5000)
endif()
