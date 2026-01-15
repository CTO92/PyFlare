# PyFlare Common Utilities Library

add_library(pyflare_common STATIC
    logging.cpp
    config.cpp
    metrics.cpp
    error.cpp
    thread_pool.cpp
)

add_library(PyFlare::Common ALIAS pyflare_common)

target_include_directories(pyflare_common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/pyflare>
)

target_link_libraries(pyflare_common
    PUBLIC
        spdlog::spdlog
        fmt::fmt
        nlohmann_json::nlohmann_json
        yaml-cpp::yaml-cpp
        absl::status
        absl::statusor
        absl::strings
        absl::time
        Threads::Threads
)

target_compile_features(pyflare_common PUBLIC cxx_std_20)
