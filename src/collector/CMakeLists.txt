# PyFlare Collector Service

# Collector library containing all collector components
add_library(pyflare_collector_lib STATIC
    collector.cpp
    otlp_receiver.cpp
    kafka_exporter.cpp
    batcher.cpp
    sampler.cpp
)

target_include_directories(pyflare_collector_lib
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/pyflare/collector>
)

target_link_libraries(pyflare_collector_lib
    PUBLIC
        PyFlare::Common
        absl::status
        absl::statusor
        absl::strings
        nlohmann_json::nlohmann_json
        yaml-cpp::yaml-cpp
        spdlog::spdlog
)

# Optional gRPC support
find_package(gRPC CONFIG QUIET)
if(gRPC_FOUND)
    target_compile_definitions(pyflare_collector_lib PUBLIC PYFLARE_HAS_GRPC)
    target_link_libraries(pyflare_collector_lib PUBLIC gRPC::grpc++)
    message(STATUS "gRPC support enabled")
else()
    message(STATUS "gRPC not found - gRPC receiver disabled")
endif()

# Optional librdkafka support
find_package(RdKafka CONFIG QUIET)
if(RdKafka_FOUND)
    target_compile_definitions(pyflare_collector_lib PUBLIC PYFLARE_HAS_RDKAFKA)
    target_link_libraries(pyflare_collector_lib PUBLIC RdKafka::rdkafka++)
    message(STATUS "librdkafka support enabled")
else()
    message(STATUS "librdkafka not found - Kafka exporter will simulate")
endif()

# Optional cpp-httplib for HTTP server
find_package(httplib CONFIG QUIET)
if(httplib_FOUND)
    target_compile_definitions(pyflare_collector_lib PUBLIC PYFLARE_HAS_HTTPLIB)
    target_link_libraries(pyflare_collector_lib PUBLIC httplib::httplib)
    message(STATUS "cpp-httplib support enabled")
else()
    message(STATUS "cpp-httplib not found - HTTP receiver disabled")
endif()

# Collector executable
add_executable(pyflare_collector main.cpp)

target_link_libraries(pyflare_collector
    PRIVATE
        pyflare_collector_lib
        CLI11::CLI11
)

# Create alias for consistent naming
add_library(PyFlare::Collector ALIAS pyflare_collector_lib)

# Install rules
install(TARGETS pyflare_collector
    RUNTIME DESTINATION bin
)

install(TARGETS pyflare_collector_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    collector.h
    types.h
    otlp_receiver.h
    kafka_exporter.h
    batcher.h
    sampler.h
    DESTINATION include/pyflare/collector
)
