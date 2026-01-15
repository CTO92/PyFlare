# PyFlare Storage Layer Library

add_library(pyflare_storage STATIC
    clickhouse/client.cpp
    qdrant/client.cpp
)

add_library(PyFlare::Storage ALIAS pyflare_storage)

target_include_directories(pyflare_storage
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/pyflare/storage>
)

target_link_libraries(pyflare_storage
    PUBLIC
        PyFlare::Common
)

# Note: In production, these would link to actual client libraries
# target_link_libraries(pyflare_storage
#     PRIVATE
#         clickhouse-cpp-lib
#         qdrant-client
# )
