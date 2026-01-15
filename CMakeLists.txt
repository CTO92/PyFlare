cmake_minimum_required(VERSION 3.20)

project(PyFlare
    VERSION 0.1.0
    DESCRIPTION "OpenTelemetry-native observability platform for AI/ML workloads"
    LANGUAGES CXX
)

# ============================================================================
# Build Options
# ============================================================================
option(PYFLARE_BUILD_TESTS "Build unit tests" ON)
option(PYFLARE_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(PYFLARE_BUILD_DOCS "Build documentation" OFF)
option(PYFLARE_ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)
option(PYFLARE_ENABLE_COVERAGE "Enable code coverage" OFF)

# ============================================================================
# C++ Standard Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Compiler Flags
# ============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wno-unused-parameter
        -Wno-missing-field-initializers
    )

    if(PYFLARE_ENABLE_SANITIZERS)
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    endif()

    if(PYFLARE_ENABLE_COVERAGE)
        add_compile_options(--coverage -fprofile-arcs -ftest-coverage)
        add_link_options(--coverage)
    endif()
elseif(MSVC)
    add_compile_options(
        /W4
        /WX
        /permissive-
        /Zc:__cplusplus
        /utf-8
    )
endif()

# ============================================================================
# Build Type Configuration
# ============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# ============================================================================
# Output Directories
# ============================================================================
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# Dependencies
# ============================================================================

# Check if vcpkg is being used (via toolchain file)
if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg")
    set(PYFLARE_USE_VCPKG ON)
    message(STATUS "Using vcpkg for dependency management")
else()
    set(PYFLARE_USE_VCPKG OFF)
    message(STATUS "Using FetchContent for dependency management")
endif()

if(PYFLARE_USE_VCPKG)
    # Use vcpkg packages
    find_package(spdlog CONFIG REQUIRED)
    find_package(fmt CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(yaml-cpp CONFIG REQUIRED)
    find_package(absl CONFIG REQUIRED)
    find_package(CLI11 CONFIG REQUIRED)
else()
    # Fallback to FetchContent for development without vcpkg
    include(FetchContent)

    # spdlog - Logging
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.13.0
    )

    # fmt - Formatting
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 10.2.0
    )

    # nlohmann_json - JSON
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )

    # yaml-cpp - YAML configuration
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
    )

    # abseil-cpp - Google utilities
    FetchContent_Declare(
        abseil-cpp
        GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
        GIT_TAG 20240116.1
    )

    # CLI11 - Command line parsing
    FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.4.1
    )

    # Make dependencies available
    FetchContent_MakeAvailable(spdlog fmt nlohmann_json yaml-cpp abseil-cpp CLI11)
endif()

# ============================================================================
# Find System Dependencies
# ============================================================================
find_package(Threads REQUIRED)

# gRPC and Protobuf
find_package(protobuf CONFIG)
find_package(gRPC CONFIG)

# librdkafka
find_package(RdKafka CONFIG)

# ClickHouse C++ client
find_package(clickhouse-cpp CONFIG)

# OpenTelemetry C++
find_package(opentelemetry-cpp CONFIG)

# Report found dependencies
message(STATUS "")
message(STATUS "Optional Dependencies Found:")
message(STATUS "  protobuf:          ${protobuf_FOUND}")
message(STATUS "  gRPC:              ${gRPC_FOUND}")
message(STATUS "  RdKafka:           ${RdKafka_FOUND}")
message(STATUS "  clickhouse-cpp:    ${clickhouse-cpp_FOUND}")
message(STATUS "  opentelemetry-cpp: ${opentelemetry-cpp_FOUND}")

# ============================================================================
# PyFlare Libraries and Executables
# ============================================================================

# Common utilities library
add_subdirectory(src/common)

# Storage layer
add_subdirectory(src/storage)

# Stream processors
add_subdirectory(src/processor)

# Query API
add_subdirectory(src/query)

# Collector service
add_subdirectory(src/collector)

# ============================================================================
# Testing
# ============================================================================
if(PYFLARE_BUILD_TESTS)
    enable_testing()

    # Google Test
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # Prevent overriding parent project's compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    include(GoogleTest)

    add_subdirectory(tests/unit)
    add_subdirectory(tests/integration)
endif()

# ============================================================================
# Installation
# ============================================================================
include(GNUInstallDirs)

install(TARGETS pyflare_common pyflare_storage pyflare_processor pyflare_query pyflare_collector
    EXPORT PyFlareTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT PyFlareTargets
    FILE PyFlareTargets.cmake
    NAMESPACE PyFlare::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PyFlare
)

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "PyFlare ${PROJECT_VERSION} Configuration Summary")
message(STATUS "================================")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build tests:       ${PYFLARE_BUILD_TESTS}")
message(STATUS "  Build benchmarks:  ${PYFLARE_BUILD_BENCHMARKS}")
message(STATUS "  Enable sanitizers: ${PYFLARE_ENABLE_SANITIZERS}")
message(STATUS "  Enable coverage:   ${PYFLARE_ENABLE_COVERAGE}")
message(STATUS "")
