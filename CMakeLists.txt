cmake_minimum_required(VERSION 3.14)

project(GlobalOptimizers
    VERSION 0.1.0
    DESCRIPTION "Header-only library of global and local optimizers based on Eigen"
    LANGUAGES CXX)

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
    add_test(NAME optimizers COMMAND test_optimizers)
endif()

if (GLOBOPT_BUILD_EXAMPLES)
    add_executable(lbfgs_example examples/lbfgs_example.cpp)
    target_link_libraries(lbfgs_example PRIVATE globopt::globopt)
endif()
